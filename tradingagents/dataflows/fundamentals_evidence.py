"""Business-quality and valuation evidence from Yahoo Finance.

Deliberately much simpler than ``yfinance_earnings.py``. Earnings needs a
persisted point-in-time snapshot store because *revision history* is
inherently a time-series concept — asking about a past date requires knowing
what consensus looked like *then*, not today. Quality and valuation are
current-snapshot concepts (this quarter's ROE, today's P/E); the one
exception, ``margin_history``, reads yfinance's own multi-year
``income_stmt``, which is already-published historical accounting data, not a
moving consensus that needs a vintage store. So there is no point-in-time
branch here: a historical ``curr_date`` still gets today's fundamentals,
disclosed as a caveat in the rendered report rather than hidden.

Two unit conversions matter and are verified against live data (AAPL/KO/NVDA,
see ``quality_models.py``/``valuation_models.py`` module docstrings):

* ``returnOnEquity``/``returnOnAssets``/``operatingMargins``/``profitMargins``
  arrive as decimal fractions (0.15 = 15%) — passed through unchanged.
* ``debtToEquity`` arrives on yfinance's own ~0-200 scale, which is *not* a
  decimal fraction of anything (AAPL 78.4 means 0.78x) — divided by 100 here,
  before it reaches ``quality_models``.
* ``dividendYield`` arrives already in percentage points (2.38 = 2.38%, KO's
  real yield) — the *opposite* convention from the margin fields in the same
  ``.info`` dict — divided by 100 here to produce a decimal fraction, so every
  percentage-shaped number this analyst pair works with is consistently a
  decimal fraction by the time it reaches the scoring modules.
* A negative or missing trailing P/E is reported *missing*, not passed
  through as a negative signal value — see ``valuation_models``'s module
  docstring for why.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Any

import pandas as pd
import yfinance as yf

from .errors import NoMarketDataError
from .evidence_values import EPSILON, safe_float
from .quality_models import QualityEvidence, Value, finalize_evidence as finalize_quality
from .symbol_utils import normalize_symbol
from .valuation_models import ValuationEvidence, finalize_evidence as finalize_valuation

logger = logging.getLogger(__name__)

SOURCE = "yfinance (Yahoo Finance fundamentals)"

#: Same screening as yfinance_earnings.py's, duplicated rather than imported:
#: nothing here is earnings-specific, but promoting it to a shared module is a
#: separate refactor from this one.
_NON_OPERATING_QUOTE_TYPES = {
    "ETF", "MUTUALFUND", "INDEX", "CURRENCY", "CRYPTOCURRENCY", "FUTURE", "OPTION",
}

#: How many annual periods of income-statement history to pull for the
#: quality tier's margin-consistency signal. More than needed
#: (quality_models.MIN_CONSISTENCY_PERIODS=3) so a period or two of missing
#: data still clears the floor.
MARGIN_HISTORY_PERIODS = 6


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _safe_date(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return None


def _safe_info(ticker: yf.Ticker) -> dict[str, Any]:
    try:
        return ticker.info or {}
    except Exception as exc:  # noqa: BLE001 - identity is best-effort
        logger.debug("yfinance info unavailable for %s: %s", ticker.ticker, exc)
        return {}


def _value(raw: Any, *, unit: str = "number", currency: str | None = None,
          as_of: str | None = None, missing_reason: str = "not reported by Yahoo Finance",
          scale: float | None = None) -> Value:
    """One field as a :class:`Value`, with an optional unit-conversion divisor."""
    number = safe_float(raw)
    if number is None:
        return Value.missing(missing_reason, unit=unit, source=SOURCE)
    if scale is not None:
        number = number / scale
    return Value(value=number, unit=unit, currency=currency, source=SOURCE, as_of=as_of)


def _pe_value(raw: Any, *, currency: str | None, as_of: str | None) -> Value:
    """Trailing/forward P/E, with a negative multiple treated as missing.

    A negative P/E means negative trailing EPS -- there is no "cheap" or
    "expensive" reading of a company with no earnings to multiply, and
    passing the negative number through would score it as an extreme
    (wrong-direction) signal rather than reporting the real reason it is
    absent. See valuation_models.py's module docstring.
    """
    number = safe_float(raw)
    if number is None:
        return Value.missing("not reported by Yahoo Finance", unit="ratio", source=SOURCE)
    if number <= 0:
        return Value.missing(
            "negative or zero trailing EPS -- no P/E multiple exists to score",
            unit="ratio", source=SOURCE,
        )
    return Value(value=number, unit="ratio", currency=currency, source=SOURCE, as_of=as_of)


def _margin_history(ticker: yf.Ticker) -> tuple[list[Value], list[str]]:
    """Operating margin per available annual period, most-recent-first.

    Both rows must be present in the *same* column for that period to count;
    a period with revenue but no reported operating income is dropped rather
    than treated as a zero margin.
    """
    try:
        stmt = ticker.income_stmt
    except Exception as exc:  # noqa: BLE001 - history is enrichment, never fatal
        logger.info("yfinance income_stmt unavailable for %s: %s", ticker.ticker, exc)
        return [], []
    if stmt is None or not isinstance(stmt, pd.DataFrame) or stmt.empty:
        return [], []
    if "Total Revenue" not in stmt.index or "Operating Income" not in stmt.index:
        return [], []

    revenue_row = stmt.loc["Total Revenue"]
    income_row = stmt.loc["Operating Income"]

    values: list[Value] = []
    periods: list[str] = []
    for column in list(stmt.columns)[:MARGIN_HISTORY_PERIODS]:
        revenue_f = safe_float(revenue_row.get(column))
        income_f = safe_float(income_row.get(column))
        if revenue_f is None or income_f is None or abs(revenue_f) < EPSILON:
            continue
        period_label = _safe_date(column) or str(column)
        values.append(Value(value=income_f / revenue_f, unit="pct_dec", source=SOURCE,
                            as_of=period_label))
        periods.append(period_label)
    return values, periods


def _unsupported_reason(quote_type: str, canonical: str) -> str:
    return (
        f"{canonical} is a {quote_type.lower()}, not an operating company, so "
        "ROE, margins, leverage and per-share multiples do not describe it the "
        "way they describe a company. A fund's own composition would need a "
        "look-through, which this analyst does not perform."
    )


@lru_cache(maxsize=64)
def _shared_snapshot(symbol: str, as_of: str) -> tuple[str, dict[str, Any], str | None, bool]:
    """Fetch and cache the one ``.info`` call both evidence builders need.

    Returns ``(canonical, info, quote_type, is_non_operating)``. Cached by
    ``(symbol, as_of)`` for the life of the process: a Quality analyst and a
    Valuation analyst run as two separate graph nodes with no shared call
    stack, so without this each would independently pay for the same
    ``.info`` network round trip. Safe to cache process-wide because
    ``/agents`` runs TradingAgents as a fresh subprocess per job (see
    ystocker's ``agents.py``) — there is no long-lived process for a stale
    entry to outlive.
    """
    canonical = normalize_symbol(symbol)
    ticker = yf.Ticker(canonical)
    info = _safe_info(ticker)
    quote_type = (info.get("quoteType") or "").strip().upper() or None
    is_non_operating = quote_type in _NON_OPERATING_QUOTE_TYPES
    return canonical, info, quote_type, is_non_operating


def _known_to_yahoo(info: dict[str, Any], quote_type: str | None) -> bool:
    return bool(quote_type) or bool(info.get("shortName")) or bool(info.get("longName"))


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


def get_quality_evidence(symbol: str, curr_date: str | None = None) -> str:
    """Return normalized business-quality evidence as a JSON document."""
    import json

    evidence = build_quality_evidence(symbol, curr_date)
    return json.dumps(evidence.to_dict(), ensure_ascii=False, sort_keys=True)


def build_quality_evidence(symbol: str, curr_date: str | None = None) -> QualityEvidence:
    as_of = _safe_date(curr_date) or _utc_today().isoformat()
    canonical, info, quote_type, is_non_operating = _shared_snapshot(symbol, as_of)
    ticker = yf.Ticker(canonical)

    if is_non_operating:
        return QualityEvidence.unsupported(
            symbol, as_of, _unsupported_reason(quote_type, canonical))
    if not _known_to_yahoo(info, quote_type):
        raise NoMarketDataError(symbol, canonical, "Yahoo returned no company metadata")

    currency = info.get("financialCurrency") or info.get("currency")
    revenue = info.get("totalRevenue")
    fcf = info.get("freeCashflow")

    have_anything = any(
        info.get(k) is not None
        for k in ("returnOnEquity", "operatingMargins", "debtToEquity", "currentRatio")
    )
    if not have_anything:
        return QualityEvidence.no_coverage(
            symbol, as_of,
            f"Yahoo recognises {canonical}"
            + (f" ({info.get('shortName')})" if info.get("shortName") else "")
            + " but publishes none of return on equity, operating margin, "
            "debt-to-equity or current ratio for it.",
        )

    margin_history, margin_periods = _margin_history(ticker)

    evidence = QualityEvidence(
        symbol=symbol, as_of=as_of, company_name=info.get("longName") or info.get("shortName"),
        currency=currency,
        return_on_equity=_value(info.get("returnOnEquity"), unit="pct_dec", as_of=as_of),
        operating_margin=_value(info.get("operatingMargins"), unit="pct_dec", as_of=as_of),
        profit_margin=_value(info.get("profitMargins"), unit="pct_dec", as_of=as_of),
        return_on_assets=_value(info.get("returnOnAssets"), unit="pct_dec", as_of=as_of),
        # yfinance's debtToEquity is on its own ~0-200 scale (verified live:
        # AAPL 78.4, KO 115.5, NVDA 17.0) -- /100 makes it a true ratio.
        debt_to_equity=_value(info.get("debtToEquity"), unit="ratio", as_of=as_of, scale=100.0),
        current_ratio=_value(info.get("currentRatio"), unit="ratio", as_of=as_of),
        free_cash_flow=_value(fcf, unit="currency_large", currency=currency, as_of=as_of),
        total_revenue=_value(revenue, unit="currency_large", currency=currency, as_of=as_of),
        margin_history=margin_history,
        margin_history_periods=margin_periods,
        sources=[SOURCE],
    )
    return finalize_quality(evidence)


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------


def get_valuation_evidence(symbol: str, curr_date: str | None = None) -> str:
    """Return normalized valuation evidence as a JSON document."""
    import json

    evidence = build_valuation_evidence(symbol, curr_date)
    return json.dumps(evidence.to_dict(), ensure_ascii=False, sort_keys=True)


def build_valuation_evidence(symbol: str, curr_date: str | None = None) -> ValuationEvidence:
    as_of = _safe_date(curr_date) or _utc_today().isoformat()
    canonical, info, quote_type, is_non_operating = _shared_snapshot(symbol, as_of)

    if is_non_operating:
        return ValuationEvidence.unsupported(
            symbol, as_of, _unsupported_reason(quote_type, canonical))
    if not _known_to_yahoo(info, quote_type):
        raise NoMarketDataError(symbol, canonical, "Yahoo returned no company metadata")

    currency = info.get("financialCurrency") or info.get("currency")
    have_anything = any(
        info.get(k) is not None
        for k in ("trailingPE", "forwardPE", "pegRatio", "priceToBook")
    )
    if not have_anything:
        return ValuationEvidence.no_coverage(
            symbol, as_of,
            f"Yahoo recognises {canonical}"
            + (f" ({info.get('shortName')})" if info.get("shortName") else "")
            + " but publishes none of trailing P/E, forward P/E, PEG or "
            "price-to-book for it.",
        )

    evidence = ValuationEvidence(
        symbol=symbol, as_of=as_of, company_name=info.get("longName") or info.get("shortName"),
        currency=currency,
        trailing_pe=_pe_value(info.get("trailingPE"), currency=currency, as_of=as_of),
        forward_pe=_pe_value(info.get("forwardPE"), currency=currency, as_of=as_of),
        peg_ratio=_value(info.get("pegRatio"), unit="ratio", as_of=as_of),
        price_to_book=_value(info.get("priceToBook"), unit="ratio", as_of=as_of),
        # yfinance's dividendYield is already percentage points (verified
        # live: KO 2.38, a real ~2.4% yield) -- the *opposite* convention from
        # the margin fields above, in the same .info dict. /100 makes it a
        # decimal fraction, consistent with every other pct_dec Value here.
        dividend_yield=_value(info.get("dividendYield"), unit="pct_dec", as_of=as_of, scale=100.0),
        market_cap=_value(info.get("marketCap"), unit="currency_large", currency=currency, as_of=as_of),
        sources=[SOURCE],
    )
    return finalize_valuation(evidence)

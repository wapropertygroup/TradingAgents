"""Earnings-estimate evidence from Yahoo Finance.

Yahoo publishes five tables that together make a revision-momentum read
possible, and the exact shapes matter enough to record here (verified live
against yfinance 1.6.0):

``calendar``          dict; ``Earnings Date`` is a *list* of dates — one when
                      the issuer has confirmed, two when Yahoo is guessing a
                      window. ``Earnings High/Low/Average`` may all be ``None``.
``earnings_estimate``  index ``0q/+1q/0y/+1y``; ``avg low high yearAgoEps
                      numberOfAnalysts growth currency``.
``revenue_estimate``   same index; ``avg low high numberOfAnalysts
                      yearAgoRevenue growth currency``, in absolute units.
``eps_trend``          same index; ``current 7daysAgo 30daysAgo 60daysAgo
                      90daysAgo``. This is the revision history.
``eps_revisions``      same index; ``upLast7days upLast30days downLast30days
                      downLast7Days``. Note the inconsistent capitalisation of
                      the final ``Days`` — a case-sensitive lookup silently
                      loses the 7-day down count and reports one-sided breadth.
``earnings_history``   index is the **fiscal quarter end**, not the
                      announcement date; ``epsActual epsEstimate epsDifference
                      surprisePercent``.

Four things this module refuses to do.

**It ignores Yahoo's ``growth`` column.** On a negative EPS the sign is wrong:
RIVN's FY26 consensus improving from -2.61 to -2.44 is published as ``growth:
0.2383``, and the same field on a name whose loss is *widening* is also
positive. All change arithmetic goes through
:func:`~.earnings_models.symmetric_change` instead.

**It will not answer a question about the past from today's data.** Yahoo's
lookbacks are relative to *now*, so asked on a later date they describe a later
window. A historical ``curr_date`` is served from the point-in-time store or not
at all. Nothing here reconstructs a vintage.

**It will not anchor post-earnings drift to a fiscal quarter end.** The June
quarter is announced in late July, so measuring "the reaction" from 30 June
prices reports three weeks of unrelated trading as the earnings move. Drift
needs real announcement dates, and when Yahoo's earnings-date endpoint is
unavailable — it is an HTML scrape, unlike the rest — drift is reported
unavailable.

**It refuses symbols it cannot resolve rather than answering emptily.** A bare
six-digit A-share code is rejected by string inspection with no network call,
and an unknown ticker raises :class:`NoMarketDataError`, so both fall through to
the next vendor in the chain. An instrument Yahoo *does* know but which has no
earnings (ETF, index, FX, crypto) is a real answer — ``unsupported`` — not a
routing miss, because no later vendor would do better.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from .earnings_models import (
    PERIOD_CURRENT_YEAR,
    PERIOD_NEXT_YEAR,
    DriftObservation,
    EarningsCalendar,
    EarningsEvidence,
    EstimateTrend,
    FiscalPeriod,
    PeriodEvidence,
    RevisionBreadth,
    SurpriseEvent,
    Value,
    finalize_evidence,
    resolve_annual_period_end,
    safe_date,
    safe_float,
    safe_int,
)
from .errors import NoMarketDataError
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

SOURCE = "yfinance (Yahoo Finance analyst estimates)"

#: Drift horizons in trading sessions.
DRIFT_HORIZONS = (1, 5, 20, 60)

#: How many announcements to measure drift for. ``earnings_history`` returns
#: four quarters, and each event costs a slice of an already-loaded price frame.
MAX_DRIFT_EVENTS = 4

#: Quote types that have no earnings. Answering ``unsupported`` for these is a
#: measurement, not a failure.
_NON_OPERATING_QUOTE_TYPES = {
    "ETF", "MUTUALFUND", "INDEX", "CURRENCY", "CRYPTOCURRENCY", "FUTURE", "OPTION",
}

#: Purely numeric symbols are not Yahoo equities — every numeric venue
#: (Shanghai, Tokyo, Hong Kong) carries a suffix. Refusing these by inspection
#: keeps a bare A-share code from costing a round trip before it reaches the
#: A-share vendor.
_BARE_NUMERIC = re.compile(r"^\d+$")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def get_earnings_evidence(symbol: str, curr_date: str | None = None) -> str:
    """Return normalized earnings evidence as a JSON document.

    JSON rather than prose because every number in the published report is
    computed from these fields by code; handing the analyst a formatted table
    would make the figures re-transcribable, and a transcription error in a
    consensus estimate is indistinguishable from a revision.
    """
    import json

    evidence = build_earnings_evidence(symbol, curr_date)
    return json.dumps(evidence.to_dict(), ensure_ascii=False, sort_keys=True)


def build_earnings_evidence(symbol: str, curr_date: str | None = None) -> EarningsEvidence:
    """Assemble :class:`EarningsEvidence` for ``symbol`` as of ``curr_date``."""
    as_of = safe_date(curr_date) or _utc_today().isoformat()

    if _BARE_NUMERIC.match((symbol or "").strip()):
        raise NoMarketDataError(
            symbol, None,
            "bare numeric symbol is not a Yahoo ticker; a venue suffix is required "
            "(e.g. 600519.SS, 7203.T, 0700.HK)",
        )

    canonical = normalize_symbol(symbol)

    # Instruments whose *symbol form* already settles it, with no network call.
    unsupported = _unsupported_by_symbol_form(canonical)
    if unsupported is not None:
        return EarningsEvidence.unsupported(symbol, as_of, unsupported)

    if _is_historical(as_of):
        return _serve_point_in_time(symbol, canonical, as_of)

    return _fetch_live(symbol, canonical, as_of)


# ---------------------------------------------------------------------------
# Date handling
# ---------------------------------------------------------------------------


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _is_historical(as_of: str) -> bool:
    """True when ``as_of`` precedes today under *both* UTC and local reckoning.

    A run started at 23:30 in UTC-8 is "today" locally and "tomorrow" in UTC;
    a run at 07:00 in UTC+13 is the reverse. Requiring the date to be behind the
    earlier of the two means a legitimate same-day run is never diverted onto
    the point-in-time path by a timezone offset.

    Erring toward "live" is safe only because the *write* is keyed by the real
    observation date, never by ``as_of`` — see :func:`_fetch_live`. Without that
    separation this bias would let a boundary case file today's consensus as
    yesterday's vintage, which is the one error the store exists to prevent.
    """
    try:
        requested = date.fromisoformat(as_of)
    except ValueError:
        return False
    return requested < min(_utc_today(), datetime.now().date())


def _serve_point_in_time(symbol: str, canonical: str, as_of: str) -> EarningsEvidence:
    """Read a stored observation, or admit there is none.

    No live call is attempted and no already-public history is grafted on. The
    tempting exception — "surprise history is public, include it" — does not
    survive contact with the dates: ``earnings_history`` is indexed by fiscal
    quarter end, so a row for the June quarter looks eligible on 5 July while
    the figures were not announced until 31 July. Filtering it correctly needs
    announcement dates, which are exactly what is missing, and filtering it with
    an assumed reporting lag would put a guess in the look-ahead guard.
    """
    from .earnings_snapshot_store import SnapshotStoreError, default_store

    try:
        stored = default_store().latest_at_or_before(canonical, as_of)
    except SnapshotStoreError as exc:
        logger.warning("earnings point-in-time store unavailable for %s: %s", symbol, exc)
        return EarningsEvidence.pit_unavailable(
            symbol, as_of,
            f"The point-in-time snapshot store could not be read ({exc}). Historical "
            "estimate evidence is unavailable; today's consensus is not a substitute.",
        )

    if stored is None:
        return EarningsEvidence.pit_unavailable(
            symbol, as_of,
            f"No estimate snapshot was observed on or before {as_of}. Yahoo's "
            "revision lookbacks are relative to the present, so today's figures "
            "cannot describe that date. Snapshots accumulate one per day from the "
            "first run onward; a historical date before the first run has no "
            "vintage and this run reports none.",
        )

    return replace(stored, symbol=symbol, as_of=as_of)


# ---------------------------------------------------------------------------
# Symbol-form screening
# ---------------------------------------------------------------------------


def _unsupported_by_symbol_form(canonical: str) -> str | None:
    """Name the reason a symbol form cannot have earnings, or ``None``."""
    if canonical.startswith("^"):
        return f"{canonical} is a market index, which does not report earnings."
    if canonical.endswith("=X"):
        return f"{canonical} is a foreign-exchange pair, which does not report earnings."
    if canonical.endswith("=F"):
        return f"{canonical} is a futures contract, which does not report earnings."
    if canonical.endswith("-USD"):
        return (
            f"{canonical} is a crypto asset. It has no issuer, no analyst consensus "
            "and no earnings calendar."
        )
    return None


# ---------------------------------------------------------------------------
# Live fetch
# ---------------------------------------------------------------------------


def _fetch_live(symbol: str, canonical: str, as_of: str) -> EarningsEvidence:
    from .stockstats_utils import yf_retry

    ticker = yf.Ticker(canonical)
    info = _safe_info(ticker)
    quote_type = (info.get("quoteType") or "").strip().upper() or None

    if quote_type in _NON_OPERATING_QUOTE_TYPES:
        return EarningsEvidence.unsupported(
            symbol, as_of,
            f"{canonical} is a {quote_type.lower()}, not an operating company, so it "
            "has no analyst EPS consensus, revision history or earnings calendar. "
            "Holdings-level earnings would need a look-through, which this analyst "
            "does not perform.",
        )

    estimates = _table(ticker, "earnings_estimate", yf_retry)
    revenue = _table(ticker, "revenue_estimate", yf_retry)
    trend = _table(ticker, "eps_trend", yf_retry)
    revisions = _table(ticker, "eps_revisions", yf_retry)
    history = _table(ticker, "earnings_history", yf_retry)
    calendar_raw = _calendar(ticker, yf_retry)

    known_to_yahoo = bool(quote_type) or bool(info.get("shortName")) or bool(info.get("longName"))
    have_anything = any(
        frame is not None and not frame.empty
        for frame in (estimates, revenue, trend, revisions, history)
    )

    if not known_to_yahoo and not have_anything:
        # Yahoo has never heard of this symbol. A typed no-data error lets the
        # router try the next vendor — which is how a bare A-share code that
        # slipped past the syntactic screen still reaches the A-share vendor.
        raise NoMarketDataError(
            symbol, canonical,
            "Yahoo returned no company metadata and no estimate tables",
        )

    if not have_anything:
        return EarningsEvidence.no_coverage(
            symbol, as_of,
            f"Yahoo recognises {canonical}"
            + (f" ({info.get('shortName')})" if info.get("shortName") else "")
            + " but publishes no analyst EPS estimates, revision history, or "
            "reported-quarter history for it. This is an absence of sell-side "
            "coverage, not a zero: no revision direction can be inferred.",
        )

    next_fye = _epoch_to_date(info.get("nextFiscalYearEnd"))
    periods = _build_periods(estimates, revenue, trend, revisions, next_fye)

    evidence = EarningsEvidence(
        symbol=symbol,
        as_of=as_of,
        canonical_symbol=canonical,
        company_name=info.get("longName") or info.get("shortName"),
        currency=info.get("financialCurrency") or info.get("currency"),
        quote_type=quote_type,
        periods=periods,
        calendar=_build_calendar(calendar_raw, info),
        surprises=_build_surprises(history),
        sources=[SOURCE],
    )

    evidence = _attach_revenue_trend_from_snapshots(evidence, canonical)
    evidence = _attach_drift(evidence, ticker, canonical, as_of)
    evidence = _attach_notes(evidence, next_fye, as_of)
    evidence = finalize_evidence(evidence)

    _persist(evidence, canonical)
    return evidence


def _safe_info(ticker: yf.Ticker) -> dict[str, Any]:
    """``ticker.info`` or ``{}``. Never fatal: the tables are the payload."""
    try:
        return ticker.info or {}
    except Exception as exc:  # noqa: BLE001 - identity is best-effort
        logger.debug("yfinance info unavailable for %s: %s", ticker.ticker, exc)
        return {}


def _table(ticker: yf.Ticker, attribute: str, retry) -> pd.DataFrame | None:
    """One estimate table, or ``None`` when Yahoo does not serve it.

    A single missing table must not sink the request: a company can have an EPS
    trend and no revenue estimate, and the report says which is which.
    """
    try:
        frame = retry(lambda: getattr(ticker, attribute))
    except Exception as exc:  # noqa: BLE001 - per-table degradation
        logger.info("yfinance %s unavailable for %s: %s", attribute, ticker.ticker, exc)
        return None
    if frame is None or not isinstance(frame, pd.DataFrame):
        return None
    return frame


def _calendar(ticker: yf.Ticker, retry) -> dict[str, Any]:
    try:
        raw = retry(lambda: ticker.calendar)
    except Exception as exc:  # noqa: BLE001
        logger.info("yfinance calendar unavailable for %s: %s", ticker.ticker, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def _epoch_to_date(raw: Any) -> str | None:
    """Yahoo returns fiscal year ends as epoch seconds."""
    seconds = safe_float(raw)
    if seconds is None or seconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Table normalization
# ---------------------------------------------------------------------------


def _cell(frame: pd.DataFrame | None, row: str, *column_aliases: str) -> Any:
    """Read one cell, matching the column name case-insensitively.

    The case-insensitive match is not defensive programming for its own sake:
    ``eps_revisions`` ships ``upLast7days`` beside ``downLast7Days``, so a
    literal lookup for one style silently returns nothing for the other and the
    7-day breadth is reported as up-only.
    """
    if frame is None or frame.empty or row not in frame.index:
        return None
    lowered = {str(c).lower(): c for c in frame.columns}
    for alias in column_aliases:
        column = lowered.get(alias.lower())
        if column is None:
            continue
        try:
            return frame.loc[row, column]
        except (KeyError, IndexError):
            continue
    return None


def _currency_for(frame: pd.DataFrame | None, row: str) -> str | None:
    raw = _cell(frame, row, "currency")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _value(
    raw: Any,
    *,
    unit: str = "number",
    currency: str | None = None,
    as_of: str | None = None,
    missing_reason: str = "not reported by Yahoo Finance",
) -> Value:
    number = safe_float(raw)
    if number is None:
        return Value.missing(missing_reason, unit=unit, source=SOURCE)
    return Value(
        value=number, unit=unit, currency=currency, source=SOURCE, as_of=as_of
    )


def _count(raw: Any, *, as_of: str | None = None, missing_reason: str) -> Value:
    number = safe_int(raw)
    if number is None:
        return Value.missing(missing_reason, unit="count", source=SOURCE)
    return Value(value=float(number), unit="count", source=SOURCE, as_of=as_of)


def _build_periods(
    estimates: pd.DataFrame | None,
    revenue: pd.DataFrame | None,
    trend: pd.DataFrame | None,
    revisions: pd.DataFrame | None,
    next_fiscal_year_end: str | None,
) -> dict[str, PeriodEvidence]:
    """Assemble one :class:`PeriodEvidence` per relative period Yahoo returns."""
    keys: list[str] = []
    for frame in (estimates, trend, revisions, revenue):
        if frame is None or frame.empty:
            continue
        for key in frame.index:
            text = str(key).strip()
            # ``growth_estimates`` carries an ``LTG`` row; the estimate tables do
            # not, but guard anyway — a long-term-growth row has no horizons and
            # would render as a period with every field unavailable.
            if text and text.upper() != "LTG" and text not in keys:
                keys.append(text)

    periods: dict[str, PeriodEvidence] = {}
    for key in keys:
        currency = (
            _currency_for(estimates, key)
            or _currency_for(trend, key)
            or _currency_for(revenue, key)
        )
        end_date = resolve_annual_period_end(
            key, next_fiscal_year_end=next_fiscal_year_end
        )

        # ``eps_trend.current`` and ``earnings_estimate.avg`` are the same
        # consensus from two endpoints. Prefer the trend table so the "today"
        # figure and its own lookbacks come from one payload — mixing them can
        # show a change that is a table-refresh skew rather than a revision.
        current_raw = _cell(trend, key, "current")
        if current_raw is None:
            current_raw = _cell(estimates, key, "avg")

        eps = EstimateTrend(
            current=_value(
                current_raw,
                currency=currency,
                missing_reason="Yahoo published no current EPS consensus for this period",
            ),
            days_ago_7=_value(_cell(trend, key, "7daysAgo"), currency=currency),
            days_ago_30=_value(_cell(trend, key, "30daysAgo"), currency=currency),
            days_ago_60=_value(_cell(trend, key, "60daysAgo"), currency=currency),
            days_ago_90=_value(_cell(trend, key, "90daysAgo"), currency=currency),
        )

        revenue_trend = EstimateTrend(
            current=_value(
                _cell(revenue, key, "avg"),
                unit="currency_large",
                currency=currency,
            ),
            # Yahoo publishes no revenue revision history at any horizon. These
            # stay missing with a reason rather than being back-filled from the
            # EPS horizons, which would silently duplicate the EPS signal into
            # the revenue weight and double-count it.
            days_ago_7=Value.missing(
                "Yahoo publishes no revenue revision history", source=SOURCE
            ),
            days_ago_30=Value.missing(
                "Yahoo publishes no revenue revision history", source=SOURCE
            ),
            days_ago_60=Value.missing(
                "Yahoo publishes no revenue revision history", source=SOURCE
            ),
            days_ago_90=Value.missing(
                "Yahoo publishes no revenue revision history", source=SOURCE
            ),
        )

        breadth = RevisionBreadth(
            up_7d=_count(
                _cell(revisions, key, "upLast7days", "upLast7Days"),
                missing_reason="not reported by Yahoo Finance",
            ),
            down_7d=_count(
                _cell(revisions, key, "downLast7Days", "downLast7days"),
                missing_reason="not reported by Yahoo Finance",
            ),
            up_30d=_count(
                _cell(revisions, key, "upLast30days", "upLast30Days"),
                missing_reason="not reported by Yahoo Finance",
            ),
            down_30d=_count(
                _cell(revisions, key, "downLast30Days", "downLast30days"),
                missing_reason="not reported by Yahoo Finance",
            ),
            # Yahoo gives a 90-day *trend* but only 7- and 30-day *counts*.
            # Reusing the 30-day counts here would read as a measurement.
            up_90d=Value.missing(
                "Yahoo publishes 90-day EPS trend but no 90-day revision counts",
                unit="count", source=SOURCE,
            ),
            down_90d=Value.missing(
                "Yahoo publishes 90-day EPS trend but no 90-day revision counts",
                unit="count", source=SOURCE,
            ),
        )

        periods[key] = PeriodEvidence(
            period=FiscalPeriod(key=key, end_date=end_date),
            eps=eps,
            revenue=revenue_trend,
            breadth=breadth,
            analyst_count=_count(
                _cell(estimates, key, "numberOfAnalysts")
                if estimates is not None
                else None,
                missing_reason="analyst coverage count not reported",
            ),
            year_ago_eps=_value(_cell(estimates, key, "yearAgoEps"), currency=currency),
        )
    return periods


def _build_calendar(raw: dict[str, Any], info: dict[str, Any]) -> EarningsCalendar:
    """Normalize the calendar dict, preserving an unconfirmed date range.

    ``Earnings Date`` is a list. Two entries means Yahoo is describing a window
    the issuer has not confirmed, and collapsing it to the first date would
    present a guess as a schedule — which is how an earnings-blackout rule
    passes on the day of a release.
    """
    dates = raw.get("Earnings Date") or []
    if not isinstance(dates, (list, tuple)):
        dates = [dates]
    parsed = [d for d in (safe_date(d) for d in dates) if d is not None]
    parsed.sort()

    if not parsed:
        return EarningsCalendar(
            unavailable_reason=(
                "Yahoo returned no upcoming earnings date. It may be unscheduled, or "
                "the issuer may not have confirmed one."
            )
        )

    currency = info.get("financialCurrency") or info.get("currency")
    return EarningsCalendar(
        next_date=parsed[0],
        next_date_range_end=parsed[-1] if len(parsed) > 1 else None,
        date_is_estimated=len(parsed) > 1,
        # Yahoo's calendar carries no before/after-market flag. Left as
        # ``unknown`` rather than assumed: an after-close release moves the
        # *next* session, and guessing inverts the drift anchor by a day.
        timing="unknown",
        eps_estimate_avg=_value(raw.get("Earnings Average"), currency=currency),
        eps_estimate_low=_value(raw.get("Earnings Low"), currency=currency),
        eps_estimate_high=_value(raw.get("Earnings High"), currency=currency),
        revenue_estimate_avg=_value(
            raw.get("Revenue Average"), unit="currency_large", currency=currency
        ),
    )


def _build_surprises(history: pd.DataFrame | None) -> list[SurpriseEvent]:
    if history is None or history.empty:
        return []
    events: list[SurpriseEvent] = []
    for row in history.index:
        quarter = safe_date(row)
        if quarter is None:
            continue
        events.append(
            SurpriseEvent(
                fiscal_period_end=quarter,
                # Deliberately absent: ``earnings_history`` has no announcement
                # date, and the index is the quarter end. Filled in by
                # :func:`_attach_drift` only when the earnings-date endpoint
                # supplies a real one.
                announcement_date=None,
                eps_actual=_value(_cell(history, row, "epsActual")),
                eps_estimate=_value(_cell(history, row, "epsEstimate")),
                eps_difference=_value(_cell(history, row, "epsDifference")),
                surprise_pct=_value(
                    _cell(history, row, "surprisePercent"), unit="pct_dec"
                ),
            )
        )
    events.sort(key=lambda e: e.fiscal_period_end)
    return events


# ---------------------------------------------------------------------------
# Revenue revisions from local vintages
# ---------------------------------------------------------------------------


def _attach_revenue_trend_from_snapshots(
    evidence: EarningsEvidence, canonical: str
) -> EarningsEvidence:
    """Fill lookbacks Yahoo does not publish from this installation's vintages.

    Yahoo gives an EPS trend but **no revenue revision history at any horizon**
    and no 90-day breadth, so the only honest source for a revenue revision is a
    snapshot this installation took itself. Delegated to the shared helper so the
    A-share path — which has no history for EPS either — gets identical
    semantics: vendor-published horizons are never overwritten, each filled slot
    carries its true age, and a vintage too old for its slot is refused.
    """
    from .earnings_snapshot_store import (
        SnapshotStoreError,
        backfill_trend_from_snapshots,
    )

    try:
        return backfill_trend_from_snapshots(evidence, canonical)
    except SnapshotStoreError as exc:
        logger.info("earnings snapshot backfill failed for %s: %s", canonical, exc)
        return evidence


# ---------------------------------------------------------------------------
# Post-earnings drift
# ---------------------------------------------------------------------------


def _attach_drift(
    evidence: EarningsEvidence, ticker: yf.Ticker, canonical: str, as_of: str
) -> EarningsEvidence:
    """Measure drift from real announcement dates, or say why not."""
    announcements = _announcement_dates(ticker, as_of)
    if not announcements:
        return replace(
            evidence,
            drift_unavailable_reason=(
                "Yahoo's earnings-date endpoint returned no usable announcement dates. "
                "It is an HTML scrape rather than a JSON API, so it fails "
                "independently of the estimate tables. Drift is not estimated from "
                "fiscal quarter ends: a June quarter is announced in late July, so "
                "that anchor would report weeks of unrelated trading as the earnings "
                "reaction."
            ),
        )

    # Stamp announcement dates onto the surprise rows they belong to, matching
    # each announcement to the most recent quarter end that precedes it.
    surprises = _match_announcements(evidence.surprises, announcements)

    try:
        prices = _adjusted_closes(canonical, as_of)
    except Exception as exc:  # noqa: BLE001 - drift is context, never fatal
        logger.info("drift price history unavailable for %s: %s", canonical, exc)
        return replace(
            evidence,
            surprises=surprises,
            drift_unavailable_reason=f"adjusted price history unavailable ({exc})",
        )

    benchmark_symbol = _benchmark_for(canonical)
    benchmark_prices = None
    if benchmark_symbol and benchmark_symbol.upper() != canonical.upper():
        try:
            benchmark_prices = _adjusted_closes(benchmark_symbol, as_of)
        except Exception as exc:  # noqa: BLE001 - excess return is optional
            logger.info("benchmark %s unavailable for drift: %s", benchmark_symbol, exc)

    observations = compute_drift_windows(
        prices,
        [(s.fiscal_period_end, s.announcement_date) for s in surprises if s.announcement_date],
        benchmark=benchmark_prices,
        horizons=DRIFT_HORIZONS,
        max_events=MAX_DRIFT_EVENTS,
    )
    if not observations:
        return replace(
            evidence,
            surprises=surprises,
            drift_unavailable_reason=(
                "announcement dates were resolved but the price history did not cover "
                "enough sessions after them to measure any window"
            ),
        )
    return replace(evidence, surprises=surprises, drift=observations)


def _announcement_dates(ticker: yf.Ticker, as_of: str) -> list[str]:
    """Past announcement dates at or before ``as_of``, newest last.

    Wrapped broadly on purpose. ``earnings_dates`` scrapes
    ``finance.yahoo.com/calendar/earnings`` and parses HTML with BeautifulSoup,
    so it breaks on layout changes, blocks and missing optional dependencies in
    ways the JSON endpoints do not.
    """
    try:
        frame = ticker.get_earnings_dates(limit=12)
    except Exception as exc:  # noqa: BLE001
        logger.info("yfinance earnings_dates unavailable for %s: %s", ticker.ticker, exc)
        return []
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return []

    dates: list[str] = []
    for raw in frame.index:
        parsed = safe_date(raw)
        if parsed is not None and parsed <= as_of:
            dates.append(parsed)
    return sorted(set(dates))


def _match_announcements(
    surprises: list[SurpriseEvent], announcements: list[str]
) -> list[SurpriseEvent]:
    """Attach each quarter end the earliest announcement that follows it.

    A quarter is reported after it closes, so the announcement for the quarter
    ending 30 June is the first announcement dated after 30 June. Matching to
    the *nearest* date in either direction would attach the previous quarter's
    release, since that is usually closer.
    """
    out: list[SurpriseEvent] = []
    for event in surprises:
        later = [a for a in announcements if a > event.fiscal_period_end]
        out.append(
            replace(event, announcement_date=later[0]) if later else event
        )
    return out


def _adjusted_closes(symbol: str, as_of: str) -> pd.DataFrame:
    """Split/dividend-adjusted closes up to ``as_of``, from the shared cache.

    Reuses ``load_ohlcv``, which caches five years per symbol, fetches with
    ``auto_adjust=True`` and filters to ``as_of``. So drift costs no extra
    network call once the market analyst has run, and cannot see a price the
    requested date could not.
    """
    from .stockstats_utils import load_ohlcv

    frame = load_ohlcv(symbol, as_of)
    if frame is None or frame.empty or "Close" not in frame.columns:
        raise NoMarketDataError(symbol, symbol, "no adjusted closes available")
    return frame[["Date", "Close"]].dropna().sort_values("Date").reset_index(drop=True)


def _benchmark_for(canonical: str) -> str | None:
    """The configured benchmark for excess return, by exchange suffix."""
    from .config import get_config

    config = get_config()
    explicit = config.get("benchmark_ticker")
    if explicit:
        return str(explicit)
    mapping = config.get("benchmark_map") or {}
    upper = canonical.upper()
    for suffix, benchmark in mapping.items():
        if suffix and upper.endswith(str(suffix).upper()):
            return str(benchmark)
    return str(mapping.get("", "SPY")) or None


def compute_drift_windows(
    prices: pd.DataFrame,
    events: list[tuple[str, str]],
    *,
    benchmark: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = DRIFT_HORIZONS,
    max_events: int = MAX_DRIFT_EVENTS,
) -> list[DriftObservation]:
    """Pure drift arithmetic over a price frame.

    ``prices`` needs ``Date`` and ``Close`` columns; ``events`` is
    ``(fiscal_period_end, announcement_date)`` pairs. No I/O here, so every
    calendar edge — an after-close release, a holiday, a Friday announcement,
    a window that runs off the end of the data — is testable from a fixture.

    The anchor is the first session **strictly after** the announcement date.
    That is the conservative reading of an unknown release time: a company that
    reported after the close moved the next session, and a company that reported
    before the open moved the announcement session itself. Anchoring to the
    following session understates a before-the-open release by one day; the
    alternative overstates an after-the-close release by including a session
    that closed before the news existed.
    """
    if prices is None or prices.empty:
        return []

    dates = pd.to_datetime(prices["Date"]).dt.tz_localize(None)
    closes = pd.to_numeric(prices["Close"], errors="coerce")
    frame = pd.DataFrame({"Date": dates, "Close": closes}).dropna().reset_index(drop=True)
    if frame.empty:
        return []

    bench_frame = None
    if benchmark is not None and not benchmark.empty and "Close" in benchmark.columns:
        b_dates = pd.to_datetime(benchmark["Date"]).dt.tz_localize(None)
        b_closes = pd.to_numeric(benchmark["Close"], errors="coerce")
        bench_frame = (
            pd.DataFrame({"Date": b_dates, "Close": b_closes})
            .dropna()
            .reset_index(drop=True)
        )

    observations: list[DriftObservation] = []
    recent = sorted(events, key=lambda e: e[1])[-max_events:]
    for fiscal_end, announced in recent:
        stamp = pd.to_datetime(announced, errors="coerce")
        if pd.isna(stamp):
            continue
        after = frame.index[frame["Date"] > stamp]
        if len(after) == 0:
            continue
        anchor = int(after[0])
        anchor_date = frame.at[anchor, "Date"].date().isoformat()
        anchor_close = float(frame.at[anchor, "Close"])
        if anchor_close == 0:
            continue

        for sessions in horizons:
            end = anchor + sessions
            if end >= len(frame):
                # The window runs past the data. Omitted rather than truncated
                # to a shorter horizon under its original label, which would
                # report a 12-session move as the 20-session drift.
                continue
            stock_ret = float(frame.at[end, "Close"]) / anchor_close - 1.0
            end_date = frame.at[end, "Date"]
            bench_ret = _benchmark_return(bench_frame, frame.at[anchor, "Date"], end_date)
            excess = (
                Value(
                    value=stock_ret - bench_ret,
                    unit="pct_dec",
                    source=SOURCE,
                    as_of=anchor_date,
                )
                if bench_ret is not None
                else Value.missing("benchmark history unavailable", unit="pct_dec")
            )
            observations.append(
                DriftObservation(
                    fiscal_period_end=fiscal_end,
                    announcement_date=announced,
                    anchor_session=anchor_date,
                    sessions=sessions,
                    stock_return=Value(
                        value=stock_ret, unit="pct_dec", source=SOURCE, as_of=anchor_date
                    ),
                    benchmark_return=(
                        Value(
                            value=bench_ret, unit="pct_dec", source=SOURCE, as_of=anchor_date
                        )
                        if bench_ret is not None
                        else Value.missing(
                            "benchmark history unavailable", unit="pct_dec"
                        )
                    ),
                    excess_return=excess,
                )
            )
    return observations


def _benchmark_return(
    bench: pd.DataFrame | None,
    anchor_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float | None:
    """The benchmark's return over the stock's *calendar* window.

    Aligned by date, not by row offset or session count. Counting the same number
    of the benchmark's own sessions looks equivalent and is not: an instrument and
    its index do not always keep the same holidays, so N sessions on each side can
    span different calendar spans and the excess return then subtracts a return
    earned over a different period. Measured on a fixture where the benchmark is
    missing one session, session-count alignment reported 1.206% against the
    correct 1.000%.

    Each endpoint is the benchmark's last close at or before the stock's
    corresponding date, so a benchmark holiday carries the previous close forward
    rather than shifting the window.
    """
    if bench is None or bench.empty:
        return None
    start_idx = bench.index[bench["Date"] <= anchor_date]
    end_idx = bench.index[bench["Date"] <= end_date]
    if len(start_idx) == 0 or len(end_idx) == 0:
        return None
    start = int(start_idx[-1])
    finish = int(end_idx[-1])
    if finish <= start:
        return None
    start_close = float(bench.at[start, "Close"])
    if start_close == 0:
        return None
    return float(bench.at[finish, "Close"]) / start_close - 1.0


# ---------------------------------------------------------------------------
# Notes and persistence
# ---------------------------------------------------------------------------


def _attach_notes(
    evidence: EarningsEvidence, next_fiscal_year_end: str | None, as_of: str
) -> EarningsEvidence:
    gaps = list(evidence.data_gaps)
    warnings = list(evidence.warnings)

    if next_fiscal_year_end is None and any(
        key in evidence.periods for key in (PERIOD_CURRENT_YEAR, PERIOD_NEXT_YEAR)
    ):
        gaps.append(
            "Fiscal year end unknown, so annual periods are labelled by Yahoo's "
            "relative keys (0y / +1y) rather than a fiscal year. No FY number is "
            "invented from the calendar year."
        )

    gaps.append(
        "Whisper expectations (buy-side / unofficial estimates) are unavailable: no "
        "free provider publishes them, and news or social sentiment cannot be "
        "converted into a numeric whisper figure."
    )
    gaps.append(
        "Consensus margin revisions are unavailable. Reported margin history and "
        "management guidance may be discussed, but neither is a consensus margin "
        "revision and must not be labelled as one."
    )

    if evidence.as_of != as_of:
        warnings.append(f"as-of mismatch: requested {as_of}, evidence carries {evidence.as_of}")

    return replace(evidence, data_gaps=gaps, warnings=warnings)


def _persist(evidence: EarningsEvidence, canonical: str) -> None:
    """Record today's observation, keyed by the real observation date.

    ``observed_date`` is the UTC date of *this fetch*, never the requested
    ``as_of``. The two are normally equal — a live fetch only happens for the
    current date — but keying on the fetch date makes it structurally impossible
    for a timezone boundary to file today's consensus as an earlier vintage,
    which would corrupt every later historical read.
    """
    from .earnings_snapshot_store import SnapshotStoreError, default_store

    try:
        default_store().append(
            evidence, observed_date=_utc_today().isoformat(), source=SOURCE
        )
    except (SnapshotStoreError, ValueError) as exc:
        # A cache that cannot be written must not fail a run that already has
        # its answer. The cost is one missing vintage, reported next time.
        logger.warning("could not persist earnings snapshot for %s: %s", canonical, exc)

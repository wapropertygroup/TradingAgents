"""Earnings evidence from Alpha Vantage, and the earnings-call transcript.

Opt-in: this vendor is only reached when a user names it in the earnings chain
*and* ``ALPHA_VANTAGE_API_KEY`` is set. With no key it raises
:class:`~.alpha_vantage_common.AlphaVantageNotConfiguredError` before any
network call, so an unconfigured install pays nothing and the router moves on.

It exists for two things yfinance cannot do.

**Real announcement dates and release timing.** ``EARNINGS`` returns
``reportedDate`` and ``reportTime`` (``pre-market`` / ``post-market``) per
quarter. That is exactly what post-earnings drift needs and what Yahoo only
exposes through an HTML scrape — so with Alpha Vantage configured, drift is
anchored to the session the news could first be traded in, and the
before/after-market assumption becomes a measurement instead of an ``unknown``.

**Earnings-call commentary.** ``EARNINGS_CALL_TRANSCRIPT`` is the only source
in this project for what management actually said, and it is quarter-addressed,
so the quarter has to be *derived from a reported date at or before the analysis
date* — asking for the current quarter returns either nothing or, worse, a call
that had not happened yet.

Two degradation rules, at different granularities on purpose:

* A failure on ``EARNINGS`` propagates. It is the core payload, and a chain that
  swallowed it would report "no earnings data" while a later vendor could have
  served the request.
* A failure on ``EARNINGS_ESTIMATES`` is caught and recorded as a data gap.
  That endpoint is premium-gated, and the free-tier notice arrives as a
  rate-limit error; letting it propagate would discard the surprise history and
  announcement dates already in hand over a field the user was never entitled
  to.

Field names are read by exact key with explicit aliases, and an absent key
becomes an unavailable value with a reason rather than a guess. That is safe
here in a way it is not for a scraped HTML table: keys like
``eps_estimate_average_30_days_ago`` are self-describing, so either the key is
present and means what it says, or it is absent and reported so.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import replace
from datetime import date, datetime, timezone
from io import StringIO
from typing import Any

from .alpha_vantage_common import (
    AlphaVantageRateLimitError,
    _make_api_request,
    get_api_key,
)
from .earnings_models import (
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
    safe_date,
    safe_float,
    safe_int,
)
from .errors import NoMarketDataError
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

SOURCE = "Alpha Vantage"

#: ``EARNINGS_ESTIMATES`` horizon strings mapped onto the relative period keys
#: this project uses everywhere else. An unrecognised horizon is skipped rather
#: than assigned a key, so a new Alpha Vantage horizon cannot silently overwrite
#: the period momentum is scored against.
_HORIZON_TO_PERIOD = {
    "current quarter": "0q",
    "next quarter": "+1q",
    "current fiscal year": "0y",
    "next fiscal year": "+1y",
}

_REPORT_TIME_TO_TIMING = {
    "pre-market": "bmo (before market open)",
    "post-market": "amc (after market close)",
}

MAX_SURPRISE_QUARTERS = 8


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def get_earnings_evidence(symbol: str, curr_date: str | None = None) -> str:
    """Return normalized Alpha Vantage earnings evidence as a JSON document."""
    return json.dumps(
        build_earnings_evidence(symbol, curr_date).to_dict(),
        ensure_ascii=False,
        sort_keys=True,
    )


def get_earnings_commentary(symbol: str, curr_date: str | None = None) -> str:
    """Return the most recent *already-published* earnings call transcript.

    The quarter is derived from ``EARNINGS``' reported dates, keeping only
    quarters whose ``reportedDate`` is at or before ``curr_date``. Requesting the
    quarter the calendar is currently in would ask for a call that has not
    happened, and Alpha Vantage answers that with an empty transcript — which
    reads downstream as "management said nothing".
    """
    get_api_key()  # raises AlphaVantageNotConfiguredError with no key configured
    canonical = normalize_symbol(symbol)
    as_of = safe_date(curr_date) or datetime.now(timezone.utc).date().isoformat()

    payload = _earnings_payload(canonical, symbol)
    quarters = _reported_quarters(payload, as_of)
    if not quarters:
        return (
            f"EARNINGS_COMMENTARY_UNAVAILABLE: no earnings call at or before {as_of} "
            f"could be identified for {canonical} from Alpha Vantage's reported "
            "dates, so no transcript was requested. Do not characterise management "
            "commentary."
        )

    fiscal_end, reported = quarters[-1]
    quarter_label = _fiscal_quarter_label(fiscal_end)
    if quarter_label is None:
        return (
            f"EARNINGS_COMMENTARY_UNAVAILABLE: could not derive a transcript quarter "
            f"label from fiscal period end {fiscal_end!r}."
        )

    try:
        raw = _make_api_request(
            "EARNINGS_CALL_TRANSCRIPT", {"symbol": canonical, "quarter": quarter_label}
        )
    except AlphaVantageRateLimitError as exc:
        # Transcripts are premium on the free tier; the notice arrives here.
        return (
            f"EARNINGS_COMMENTARY_UNAVAILABLE: Alpha Vantage declined the transcript "
            f"request for {canonical} {quarter_label} ({exc}). This is an entitlement "
            "or quota limit, not evidence about the company."
        )

    turns = _parse_transcript(raw)
    if not turns:
        return (
            f"EARNINGS_COMMENTARY_UNAVAILABLE: Alpha Vantage returned no transcript "
            f"content for {canonical} {quarter_label} (reported {reported}). It may "
            "not be published yet."
        )

    header = (
        f"# Earnings call transcript — {canonical} {quarter_label}\n"
        f"Fiscal period ending {fiscal_end}; reported {reported}.\n"
        f"Source: {SOURCE} EARNINGS_CALL_TRANSCRIPT.\n"
        "Quote only what appears below; do not summarise a call that is not here.\n"
    )
    return header + "\n" + "\n\n".join(turns)


# ---------------------------------------------------------------------------
# Evidence assembly
# ---------------------------------------------------------------------------


def build_earnings_evidence(symbol: str, curr_date: str | None = None) -> EarningsEvidence:
    get_api_key()  # fail fast, before any network call, when unconfigured
    canonical = normalize_symbol(symbol)
    as_of = safe_date(curr_date) or datetime.now(timezone.utc).date().isoformat()

    payload = _earnings_payload(canonical, symbol)
    surprises = _build_surprises(payload, as_of)

    gaps: list[str] = []
    warnings: list[str] = []

    periods, estimate_gap = _build_periods(canonical, as_of)
    if estimate_gap:
        gaps.append(estimate_gap)

    calendar, calendar_gap = _build_calendar(canonical, as_of, surprises)
    if calendar_gap:
        gaps.append(calendar_gap)

    if not periods and not surprises:
        raise NoMarketDataError(
            symbol, canonical,
            "Alpha Vantage returned neither reported-quarter history nor estimates",
        )

    gaps.append(
        "Whisper expectations are unavailable: Alpha Vantage publishes sell-side "
        "consensus only, and no free source publishes buy-side whisper numbers."
    )
    gaps.append(
        "Consensus margin revisions are unavailable. Reported margins and management "
        "guidance may be discussed but are not consensus margin revisions."
    )

    evidence = EarningsEvidence(
        symbol=symbol,
        as_of=as_of,
        canonical_symbol=canonical,
        quote_type="EQUITY",
        periods=periods,
        calendar=calendar,
        surprises=surprises,
        sources=[f"{SOURCE} EARNINGS", f"{SOURCE} EARNINGS_ESTIMATES"],
        data_gaps=gaps,
        warnings=warnings,
    )
    evidence = _attach_drift(evidence, canonical, as_of)
    # Resolve the before/after-market question Yahoo leaves ``unknown``. Done
    # after construction because it reads the same EARNINGS payload the
    # surprises came from, rather than costing another request.
    evidence = attach_report_times(evidence, payload)
    return finalize_evidence(evidence)


def _earnings_payload(canonical: str, symbol: str) -> dict[str, Any]:
    """``EARNINGS`` as a dict. A failure here propagates by design."""
    raw = _make_api_request("EARNINGS", {"symbol": canonical})
    data = _as_json(raw)
    if not isinstance(data, dict) or not data:
        raise NoMarketDataError(
            symbol, canonical, "Alpha Vantage EARNINGS returned an unusable payload"
        )
    if "quarterlyEarnings" not in data and "annualEarnings" not in data:
        raise NoMarketDataError(
            symbol, canonical,
            "Alpha Vantage EARNINGS returned no annual or quarterly earnings",
        )
    return data


def _as_json(raw: Any) -> Any:
    """``_make_api_request`` hands back text for data responses; parse it."""
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _pick(row: dict[str, Any], *keys: str) -> Any:
    """First present key, matched case-insensitively and ignoring separators.

    Alpha Vantage has shipped both ``camelCase`` and ``snake_case`` across its
    endpoints, so ``reportedEPS`` and ``reported_eps`` both have to resolve. The
    match is on the normalized name only — nothing is inferred from position, so
    an unrecognised payload yields absent values rather than shifted ones.
    """
    normalized = {str(k).lower().replace("_", "").replace("-", ""): v for k, v in row.items()}
    for key in keys:
        probe = key.lower().replace("_", "").replace("-", "")
        if probe in normalized:
            return normalized[probe]
    return None


# ---------------------------------------------------------------------------
# Surprises
# ---------------------------------------------------------------------------


def _build_surprises(payload: dict[str, Any], as_of: str) -> list[SurpriseEvent]:
    """Reported quarters whose announcement is at or before ``as_of``.

    Filtered on ``reportedDate``, not on ``fiscalDateEnding``. The distinction is
    the whole point of using this endpoint: on 5 July the June quarter has ended
    but has not been announced, and a fiscal-end filter would leak a result that
    was not yet public.
    """
    rows = payload.get("quarterlyEarnings") or []
    if not isinstance(rows, list):
        return []

    events: list[SurpriseEvent] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        fiscal_end = safe_date(_pick(row, "fiscalDateEnding", "fiscal_date_ending"))
        reported = safe_date(_pick(row, "reportedDate", "reported_date"))
        if fiscal_end is None:
            continue
        if reported is not None and reported > as_of:
            continue
        if reported is None and fiscal_end > as_of:
            # Without an announcement date the only safe test is the fiscal end,
            # and that test is not sufficient — so such a row is kept only when
            # even the quarter end is in the past, and its drift is skipped
            # below for want of an anchor.
            continue

        surprise_pct = safe_float(
            _pick(row, "surprisePercentage", "surprise_percentage")
        )
        events.append(
            SurpriseEvent(
                fiscal_period_end=fiscal_end,
                announcement_date=reported,
                eps_actual=_value(_pick(row, "reportedEPS", "reported_eps")),
                eps_estimate=_value(_pick(row, "estimatedEPS", "estimated_eps")),
                eps_difference=_value(_pick(row, "surprise")),
                # Alpha Vantage publishes this as a percentage ("3.139"), while
                # yfinance publishes a decimal fraction (0.0314). Normalized to
                # the decimal here so one renderer serves both; without the
                # division the same beat reads as 314%.
                surprise_pct=(
                    Value(
                        value=surprise_pct / 100.0,
                        unit="pct_dec",
                        source=SOURCE,
                        as_of=reported,
                    )
                    if surprise_pct is not None
                    else Value.missing("not reported", unit="pct_dec", source=SOURCE)
                ),
            )
        )
    events.sort(key=lambda e: e.fiscal_period_end)
    return events[-MAX_SURPRISE_QUARTERS:]


def _value(raw: Any, *, unit: str = "number", currency: str | None = None) -> Value:
    number = safe_float(raw)
    if number is None:
        return Value.missing("not reported by Alpha Vantage", unit=unit, source=SOURCE)
    return Value(value=number, unit=unit, currency=currency, source=SOURCE)


def _reported_quarters(payload: dict[str, Any], as_of: str) -> list[tuple[str, str]]:
    """``(fiscal_end, reported_date)`` for announced quarters, oldest first."""
    out: list[tuple[str, str]] = []
    for row in payload.get("quarterlyEarnings") or []:
        if not isinstance(row, dict):
            continue
        fiscal_end = safe_date(_pick(row, "fiscalDateEnding", "fiscal_date_ending"))
        reported = safe_date(_pick(row, "reportedDate", "reported_date"))
        if fiscal_end and reported and reported <= as_of:
            out.append((fiscal_end, reported))
    out.sort(key=lambda pair: pair[1])
    return out


def _fiscal_quarter_label(fiscal_end: str) -> str | None:
    """``2024-09-30`` -> ``2024Q3``, the form the transcript endpoint takes."""
    try:
        parsed = date.fromisoformat(fiscal_end)
    except ValueError:
        return None
    return f"{parsed.year}Q{(parsed.month - 1) // 3 + 1}"


# ---------------------------------------------------------------------------
# Estimates and revisions
# ---------------------------------------------------------------------------


def _build_periods(canonical: str, as_of: str) -> tuple[dict[str, PeriodEvidence], str | None]:
    """``EARNINGS_ESTIMATES``, or a named gap. Never raises."""
    try:
        raw = _make_api_request("EARNINGS_ESTIMATES", {"symbol": canonical})
    except AlphaVantageRateLimitError as exc:
        return {}, (
            "Alpha Vantage EARNINGS_ESTIMATES was declined "
            f"({exc}). It is premium-gated, and the free tier reports that as a "
            "quota notice, so no consensus revision history is available from this "
            "vendor on this key. This is an entitlement limit, not an absence of "
            "analyst coverage."
        )
    except Exception as exc:  # noqa: BLE001 - optional enrichment
        logger.info("Alpha Vantage EARNINGS_ESTIMATES failed for %s: %s", canonical, exc)
        return {}, f"Alpha Vantage EARNINGS_ESTIMATES was unavailable ({exc})."

    data = _as_json(raw)
    rows = (data or {}).get("estimates") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        return {}, "Alpha Vantage EARNINGS_ESTIMATES returned no estimate rows."

    periods: dict[str, PeriodEvidence] = {}
    unmapped: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        horizon = str(_pick(row, "horizon") or "").strip().lower()
        key = _HORIZON_TO_PERIOD.get(horizon)
        if key is None:
            if horizon:
                unmapped.append(horizon)
            continue
        currency = _pick(row, "currency")
        currency = str(currency).strip() if currency else None
        periods[key] = PeriodEvidence(
            period=FiscalPeriod(
                key=key, end_date=safe_date(_pick(row, "date", "fiscalDateEnding"))
            ),
            eps=EstimateTrend(
                current=_value(
                    _pick(row, "eps_estimate_average", "epsEstimateAverage"),
                    currency=currency,
                ),
                days_ago_7=_value(
                    _pick(row, "eps_estimate_average_7_days_ago"), currency=currency
                ),
                days_ago_30=_value(
                    _pick(row, "eps_estimate_average_30_days_ago"), currency=currency
                ),
                days_ago_60=_value(
                    _pick(row, "eps_estimate_average_60_days_ago"), currency=currency
                ),
                days_ago_90=_value(
                    _pick(row, "eps_estimate_average_90_days_ago"), currency=currency
                ),
            ),
            revenue=EstimateTrend(
                current=_value(
                    _pick(row, "revenue_estimate_average", "revenueEstimateAverage"),
                    unit="currency_large",
                    currency=currency,
                ),
                days_ago_7=_value(
                    _pick(row, "revenue_estimate_average_7_days_ago"),
                    unit="currency_large", currency=currency,
                ),
                days_ago_30=_value(
                    _pick(row, "revenue_estimate_average_30_days_ago"),
                    unit="currency_large", currency=currency,
                ),
                days_ago_60=_value(
                    _pick(row, "revenue_estimate_average_60_days_ago"),
                    unit="currency_large", currency=currency,
                ),
                days_ago_90=_value(
                    _pick(row, "revenue_estimate_average_90_days_ago"),
                    unit="currency_large", currency=currency,
                ),
            ),
            breadth=RevisionBreadth(
                up_7d=_count(_pick(row, "eps_estimate_revision_up_trailing_7_days")),
                down_7d=_count(_pick(row, "eps_estimate_revision_down_trailing_7_days")),
                up_30d=_count(_pick(row, "eps_estimate_revision_up_trailing_30_days")),
                down_30d=_count(_pick(row, "eps_estimate_revision_down_trailing_30_days")),
                up_90d=Value.missing(
                    "Alpha Vantage publishes 7- and 30-day revision counts only",
                    unit="count", source=SOURCE,
                ),
                down_90d=Value.missing(
                    "Alpha Vantage publishes 7- and 30-day revision counts only",
                    unit="count", source=SOURCE,
                ),
            ),
            analyst_count=_count(
                _pick(row, "eps_estimate_analyst_count", "epsEstimateAnalystCount")
            ),
        )

    if not periods:
        return {}, (
            "Alpha Vantage EARNINGS_ESTIMATES returned only horizons this build does "
            f"not recognise ({', '.join(sorted(set(unmapped))) or 'none named'}), so no "
            "period was mapped. Unrecognised horizons are skipped rather than guessed."
        )
    return periods, None


def _count(raw: Any) -> Value:
    number = safe_int(raw)
    if number is None:
        return Value.missing("not reported by Alpha Vantage", unit="count", source=SOURCE)
    return Value(value=float(number), unit="count", source=SOURCE)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def _build_calendar(
    canonical: str, as_of: str, surprises: list[SurpriseEvent]
) -> tuple[EarningsCalendar, str | None]:
    """Next announcement from ``EARNINGS_CALENDAR`` (CSV), with release timing."""
    timing = _timing_from_history(surprises)
    try:
        raw = _make_api_request(
            "EARNINGS_CALENDAR", {"symbol": canonical, "horizon": "3month"}
        )
    except AlphaVantageRateLimitError as exc:
        return (
            EarningsCalendar(
                timing=timing,
                unavailable_reason=f"Alpha Vantage EARNINGS_CALENDAR declined ({exc})",
            ),
            f"Alpha Vantage EARNINGS_CALENDAR was declined ({exc}).",
        )
    except Exception as exc:  # noqa: BLE001 - optional
        logger.info("Alpha Vantage EARNINGS_CALENDAR failed for %s: %s", canonical, exc)
        return (
            EarningsCalendar(
                timing=timing,
                unavailable_reason=f"Alpha Vantage EARNINGS_CALENDAR unavailable ({exc})",
            ),
            None,
        )

    row = _first_calendar_row(raw, canonical, as_of)
    if row is None:
        return (
            EarningsCalendar(
                timing=timing,
                unavailable_reason=(
                    "Alpha Vantage's 3-month earnings calendar lists no upcoming date "
                    f"for {canonical} at or after {as_of}"
                ),
            ),
            None,
        )

    currency = row.get("currency") or None
    return (
        EarningsCalendar(
            next_date=row["reportDate"],
            # The calendar publishes a single expected date with no range and no
            # confirmation flag, so it is marked estimated: Alpha Vantage's own
            # documentation describes these as expected dates.
            date_is_estimated=True,
            timing=timing,
            eps_estimate_avg=_value(row.get("estimate"), currency=currency),
            revenue_estimate_avg=Value.missing(
                "Alpha Vantage's earnings calendar carries an EPS estimate only",
                unit="currency_large", source=SOURCE,
            ),
        ),
        None,
    )


def _first_calendar_row(raw: Any, canonical: str, as_of: str) -> dict[str, Any] | None:
    """Earliest calendar row for this symbol dated at or after ``as_of``."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        rows = list(csv.DictReader(StringIO(raw)))
    except (csv.Error, ValueError):
        return None

    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(_pick(row, "symbol") or "").strip().upper()
        if symbol and symbol != canonical.upper():
            continue
        report_date = safe_date(_pick(row, "reportDate", "report_date"))
        if report_date is None or report_date < as_of:
            continue
        candidates.append(
            {
                "reportDate": report_date,
                "fiscalDateEnding": safe_date(
                    _pick(row, "fiscalDateEnding", "fiscal_date_ending")
                ),
                "estimate": _pick(row, "estimate"),
                "currency": _pick(row, "currency"),
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda r: r["reportDate"])
    return candidates[0]


def _timing_from_history(surprises: list[SurpriseEvent]) -> str:
    """Not inferred — see :func:`_attach_report_times`; kept ``unknown`` here."""
    return "unknown"


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def _attach_drift(
    evidence: EarningsEvidence, canonical: str, as_of: str
) -> EarningsEvidence:
    """Measure drift from Alpha Vantage's real ``reportedDate`` values.

    Reuses the yfinance module's pure window arithmetic so both vendors produce
    identically-defined drift; only the announcement dates differ in origin.
    """
    from .yfinance_earnings import (
        DRIFT_HORIZONS,
        MAX_DRIFT_EVENTS,
        _adjusted_closes,
        _benchmark_for,
        compute_drift_windows,
    )

    events = [
        (s.fiscal_period_end, s.announcement_date)
        for s in evidence.surprises
        if s.announcement_date
    ]
    if not events:
        return replace(
            evidence,
            drift_unavailable_reason=(
                "no announced quarter at or before the analysis date carried a "
                "reported date, so no drift window could be anchored"
            ),
        )

    try:
        prices = _adjusted_closes(canonical, as_of)
    except Exception as exc:  # noqa: BLE001 - drift is context
        logger.info("drift price history unavailable for %s: %s", canonical, exc)
        return replace(
            evidence,
            drift_unavailable_reason=f"adjusted price history unavailable ({exc})",
        )

    benchmark = None
    bench_symbol = _benchmark_for(canonical)
    if bench_symbol and bench_symbol.upper() != canonical.upper():
        try:
            benchmark = _adjusted_closes(bench_symbol, as_of)
        except Exception as exc:  # noqa: BLE001
            logger.info("benchmark %s unavailable for drift: %s", bench_symbol, exc)

    observations: list[DriftObservation] = compute_drift_windows(
        prices,
        events,
        benchmark=benchmark,
        horizons=DRIFT_HORIZONS,
        max_events=MAX_DRIFT_EVENTS,
    )
    if not observations:
        return replace(
            evidence,
            drift_unavailable_reason=(
                "announcement dates were available but the price history did not cover "
                "enough sessions after them to measure any window"
            ),
        )
    return replace(evidence, drift=observations)


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------


def _parse_transcript(raw: Any, *, max_turns: int = 40) -> list[str]:
    """Speaker-attributed turns as markdown, capped so one call cannot flood a prompt."""
    data = _as_json(raw)
    if not isinstance(data, dict):
        return []
    turns = data.get("transcript")
    if not isinstance(turns, list):
        return []

    out: list[str] = []
    for turn in turns[:max_turns]:
        if not isinstance(turn, dict):
            continue
        content = str(_pick(turn, "content") or "").strip()
        if not content:
            continue
        speaker = str(_pick(turn, "speaker") or "Unknown speaker").strip()
        title = str(_pick(turn, "title") or "").strip()
        who = f"{speaker} — {title}" if title else speaker
        out.append(f"**{who}**\n\n{content}")
    return out


def attach_report_times(
    evidence: EarningsEvidence, payload: dict[str, Any]
) -> EarningsEvidence:
    """Record ``reportTime`` (pre/post-market) on the calendar when available.

    Split out and exported so the yfinance path can borrow it: Yahoo has no
    equivalent field, so a user with an Alpha Vantage key can resolve the
    before/after-market question Yahoo leaves ``unknown``. Uses the most recent
    announced quarter, because an issuer's release time is a stable habit while
    any single quarter could be an exception — which is why it is reported as an
    observed pattern rather than as the next release's confirmed timing.
    """
    rows = payload.get("quarterlyEarnings") or []
    latest: tuple[str, str] | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        reported = safe_date(_pick(row, "reportedDate", "reported_date"))
        report_time = str(_pick(row, "reportTime", "report_time") or "").strip().lower()
        timing = _REPORT_TIME_TO_TIMING.get(report_time)
        if reported and timing and (latest is None or reported > latest[0]):
            latest = (reported, timing)
    if latest is None:
        return evidence
    return replace(
        evidence,
        calendar=replace(
            evidence.calendar,
            timing=f"{latest[1]} — observed pattern, most recently {latest[0]}",
        ),
    )

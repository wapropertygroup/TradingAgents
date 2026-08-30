"""Earnings evidence for A-shares (沪深京) from 同花顺.

The fallback in the earnings chain, reached for bare 6-digit 沪深京 codes that
Yahoo cannot resolve. 同花顺's ``worth.html`` embeds the only free Chinese
consensus-forecast table in this project — ``a_stock.get_profit_forecast``
already scrapes it for the Fundamentals Analyst.

The table's layout was verified against the live page for 600519 and is
unambiguous, which is what makes normalizing it safe:

    年度    预测机构数   最小值   均值    最大值   行业平均数
    2026    49          64.78   67.85   77.85    8.07
    2027    48          67.23   71.79   84.02    8.64

* ``年度`` is an explicit **calendar year**, so no fiscal-year label has to be
  inferred from a relative ``0y``/``+1y`` key — the reverse of the yfinance path's
  problem.
* ``均值`` is the consensus mean EPS. It cross-checks against Yahoo's own CNY
  consensus for the same issuer (67.85 here against Yahoo's 66.86 for FY2026),
  which is the independent confirmation that the column means what its header
  says rather than being a realised historical figure.
* ``预测机构数`` is the covering-institution count, i.e. analyst coverage.

Normalization is still **conditional**: unless both a year column and a mean
column are found by exact header match, nothing is claimed numerically and the
table travels verbatim instead. A mis-mapped column here would publish a
realised result as a forward consensus at full confidence, in a report whose
whole purpose is to say which way estimates are moving.

**What is still missing, and stays missing.** The page is a *current* snapshot
with no history: no 7/30/60/90-day consensus series, no per-analyst up/down
counts, no announcement date, no surprise record. So momentum is
``Insufficient Data`` on a first run — a statement about the source, not the
company. It does not stay that way forever: every run records a dated snapshot,
and :func:`~.earnings_snapshot_store.backfill_trend_from_snapshots` fills the
lookback horizons from this installation's own accumulated vintages, so an
A-share that is analysed regularly grows a real revision history locally.

``pandas.DataFrame.to_markdown`` is deliberately not used. It requires
``tabulate``, which this project does not declare as a dependency, so the call
raises ``ImportError`` on a clean install — verified on this machine. Markdown is
assembled by hand, the way ``a_stock._sina_statement`` already does it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import date, datetime, timezone

import pandas as pd

from .a_stock import _eps_forecast_ths_frame, _historical_notice, _require_a_share
from .earnings_models import (
    EarningsCalendar,
    EarningsEvidence,
    EstimateTrend,
    FiscalPeriod,
    GuidanceNote,
    PeriodEvidence,
    RevisionBreadth,
    Value,
    finalize_evidence,
    safe_date,
    safe_float,
    safe_int,
)

logger = logging.getLogger(__name__)

SOURCE = "同花顺 (10jqka) 一致预期快照"

#: Rows carried into the verbatim note. The forecast grid is three or four rows
#: in practice; the cap stops a page-shape change from pasting a whole HTML
#: table into a prompt.
_MAX_TABLE_ROWS = 8

#: Exact header matches. Aliases are listed where 同花顺 has been seen to vary,
#: but an unrecognised header yields *no* numeric claim rather than a positional
#: guess — the columns are not a documented schema.
_YEAR_HEADERS = ("年度", "年份", "预测年度")
_MEAN_HEADERS = ("均值", "平均值", "预测均值", "一致预期")
_COUNT_HEADERS = ("预测机构数", "机构数", "预测机构家数")
_MIN_HEADERS = ("最小值", "最低值")
_MAX_HEADERS = ("最大值", "最高值")


def get_earnings_evidence(symbol: str, curr_date: str | None = None) -> str:
    """Return A-share earnings evidence as a JSON document."""
    return json.dumps(
        build_earnings_evidence(symbol, curr_date).to_dict(),
        ensure_ascii=False,
        sort_keys=True,
    )


def build_earnings_evidence(symbol: str, curr_date: str | None = None) -> EarningsEvidence:
    """Assemble what 同花顺 supports for an A-share symbol.

    Raises :class:`~.errors.NoMarketDataError` for anything that is not a 6-digit
    沪深京 code, so the router treats it as "nothing here" — the same
    refusal-based routing the rest of the A-share vendor uses.
    """
    code = _require_a_share(symbol)
    as_of = safe_date(curr_date) or datetime.now(timezone.utc).date().isoformat()

    frame = _eps_forecast_ths_frame(code)
    if frame is None or getattr(frame, "empty", True):
        return EarningsEvidence.no_coverage(
            symbol,
            as_of,
            f"同花顺 publishes no consensus-forecast table for {code}, and no other "
            "free A-share source in this project publishes analyst estimates. No "
            "EPS consensus, revision direction, or earnings calendar is available "
            "for this symbol.",
        )

    periods, mapping_note = _normalize_ths_table(frame, as_of)

    gaps = [
        "No consensus revision history (7 / 30 / 60 / 90 day) is published for "
        "A-shares by this source. Horizons shown as available were reconstructed "
        "from this installation's own dated snapshots; horizons still unavailable "
        "have no vintage yet.",
        "No analyst up/down revision counts (breadth) are available, so revision "
        "breadth cannot be reported at any window.",
        "No reported-versus-consensus surprise history is available.",
        "No post-earnings drift can be computed: this source carries no "
        "announcement dates, and drift anchored to anything else would misdate the "
        "market reaction.",
        "Whisper expectations and consensus margin revisions are unavailable.",
    ]
    if mapping_note:
        gaps.insert(0, mapping_note)

    evidence = EarningsEvidence(
        symbol=symbol,
        as_of=as_of,
        canonical_symbol=code,
        # 沪深京 issuers report in RMB. Stated rather than left blank so a
        # consensus figure is never rendered as a bare number beside a USD one.
        currency="CNY",
        quote_type="EQUITY",
        periods=periods,
        calendar=EarningsCalendar(
            unavailable_reason=(
                "同花顺's consensus page carries no scheduled announcement date. "
                "沪深京 issuers file a 预约披露时间 with their exchange, which this "
                "project does not fetch."
            )
        ),
        guidance=[
            GuidanceNote(
                text=(
                    "同花顺一致预期原始表格 / verbatim source table "
                    "(年度 = fiscal year, 预测机构数 = covering institutions, "
                    "最小值/均值/最大值 = low/mean/high EPS estimate, "
                    "行业平均数 = industry average, not this issuer):\n\n"
                    + _markdown_table(frame, _MAX_TABLE_ROWS)
                ),
                source=SOURCE,
                url=f"https://basic.10jqka.com.cn/{code}/worth.html",
            )
        ],
        sources=[SOURCE],
        data_gaps=gaps,
        drift_unavailable_reason=(
            "announcement dates are unavailable from this source, and drift anchored "
            "to anything else would misdate the market reaction"
        ),
    )

    evidence = _backfill(evidence, code)

    notice = _historical_notice(curr_date, "一致预期")
    if notice:
        # The same future-function warning the other A-share current-snapshot
        # tools carry: this is today's table, not the vintage that existed on a
        # past trade date.
        evidence = _with_warning(evidence, notice.strip())

    return finalize_evidence(evidence)


# ---------------------------------------------------------------------------
# Table normalization
# ---------------------------------------------------------------------------


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> object | None:
    """Exact header match, whitespace-insensitive. No substring or positional match."""
    normalized = {str(c).strip(): c for c in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _normalize_ths_table(
    frame: pd.DataFrame, as_of: str
) -> tuple[dict[str, PeriodEvidence], str | None]:
    """Map the forecast grid onto forecast periods.

    Returns ``({}, reason)`` when the layout is not recognised, which keeps the
    report's numeric surface free of guesses — every consumer already renders an
    absent period as unavailable.
    """
    year_col = _find_column(frame, _YEAR_HEADERS)
    mean_col = _find_column(frame, _MEAN_HEADERS)
    if year_col is None or mean_col is None:
        return {}, (
            "同花顺's forecast table did not carry the expected 年度 / 均值 columns "
            f"(saw: {', '.join(str(c) for c in frame.columns)}). Its consensus EPS is "
            "therefore NOT normalized into a measured figure — a mis-mapped column "
            "would publish a realised result as a forward consensus. Read the "
            "verbatim table under Guidance and do not restate its numbers as "
            "'consensus EPS' elsewhere."
        )

    count_col = _find_column(frame, _COUNT_HEADERS)
    min_col = _find_column(frame, _MIN_HEADERS)
    max_col = _find_column(frame, _MAX_HEADERS)

    try:
        base_year = date.fromisoformat(as_of).year
    except ValueError:
        base_year = datetime.now(timezone.utc).year

    periods: dict[str, PeriodEvidence] = {}
    for _, row in frame.iterrows():
        year = safe_int(row.get(year_col))
        mean = safe_float(row.get(mean_col))
        if year is None or mean is None or not (1990 <= year <= base_year + 10):
            continue

        offset = year - base_year
        if offset < 0:
            # A past fiscal year in a forward-estimate table is either a realised
            # figure or a stale row. Either way it is not a forecast, and treating
            # it as one is the exact failure this whole path guards against.
            continue
        key = "0y" if offset == 0 else f"+{offset}y"

        detail = []
        if min_col is not None and safe_float(row.get(min_col)) is not None:
            detail.append(f"low {safe_float(row.get(min_col))}")
        if max_col is not None and safe_float(row.get(max_col)) is not None:
            detail.append(f"high {safe_float(row.get(max_col))}")

        periods[key] = PeriodEvidence(
            period=FiscalPeriod(
                key=key,
                # 沪深京 issuers are required to use the calendar year as their
                # fiscal year, so the year end follows from 年度 with nothing
                # inferred. This is a regulatory fact, not a convention guess.
                end_date=f"{year}-12-31",
            ),
            eps=EstimateTrend(
                current=Value(
                    value=mean,
                    currency="CNY",
                    source=SOURCE + (f" ({', '.join(detail)})" if detail else ""),
                    as_of=as_of,
                ),
                # No horizons upstream. Filled from local vintages by _backfill;
                # left with an explicit reason when no vintage exists yet.
                days_ago_7=_no_history(),
                days_ago_30=_no_history(),
                days_ago_60=_no_history(),
                days_ago_90=_no_history(),
            ),
            revenue=EstimateTrend(
                current=Value.missing(
                    "同花顺's consensus table carries EPS only, no revenue estimate",
                    unit="currency_large", source=SOURCE,
                ),
                days_ago_7=_no_history(unit="currency_large"),
                days_ago_30=_no_history(unit="currency_large"),
                days_ago_60=_no_history(unit="currency_large"),
                days_ago_90=_no_history(unit="currency_large"),
            ),
            breadth=RevisionBreadth(
                up_7d=_no_breadth(), down_7d=_no_breadth(),
                up_30d=_no_breadth(), down_30d=_no_breadth(),
                up_90d=_no_breadth(), down_90d=_no_breadth(),
            ),
            analyst_count=(
                Value(
                    value=float(safe_int(row.get(count_col))),  # type: ignore[arg-type]
                    unit="count", source=SOURCE, as_of=as_of,
                )
                if count_col is not None and safe_int(row.get(count_col)) is not None
                else Value.missing(
                    "covering-institution count not reported", unit="count", source=SOURCE
                )
            ),
        )
    if not periods:
        return {}, (
            "同花顺's forecast table carried the expected columns but no row resolved "
            "to a current or future fiscal year with a usable mean estimate."
        )
    return periods, None


def _no_history(*, unit: str = "number") -> Value:
    return Value.missing(
        "同花顺 publishes a current consensus snapshot with no revision history",
        unit=unit, source=SOURCE,
    )


def _no_breadth() -> Value:
    return Value.missing(
        "同花顺 publishes no per-analyst up/down revision counts",
        unit="count", source=SOURCE,
    )


def _markdown_table(frame: pd.DataFrame, max_rows: int) -> str:
    """Hand-rolled markdown. ``to_markdown`` needs the undeclared ``tabulate``."""
    columns = [str(c).strip() for c in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.head(max_rows).iterrows():
        cells = []
        for column in frame.columns:
            value = row.get(column)
            cells.append("" if value is None or pd.isna(value) else str(value).strip())
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Local vintages
# ---------------------------------------------------------------------------


def _backfill(evidence: EarningsEvidence, code: str) -> EarningsEvidence:
    """Reconstruct lookback horizons from this installation's stored snapshots."""
    from .earnings_snapshot_store import (
        SnapshotStoreError,
        backfill_trend_from_snapshots,
        default_store,
    )

    try:
        enriched = backfill_trend_from_snapshots(evidence, code)
        default_store().append(
            enriched,
            observed_date=datetime.now(timezone.utc).date().isoformat(),
            source=SOURCE,
        )
        return enriched
    except (SnapshotStoreError, ValueError) as exc:
        logger.warning("A-share earnings snapshot unavailable for %s: %s", code, exc)
        return evidence


def _with_warning(evidence: EarningsEvidence, warning: str) -> EarningsEvidence:
    warnings = list(evidence.warnings)
    if warning not in warnings:
        warnings.append(warning)
    return replace(evidence, warnings=warnings)

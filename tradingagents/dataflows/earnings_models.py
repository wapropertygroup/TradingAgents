"""Normalized earnings-estimate evidence, and the arithmetic over it.

This module is **pure**: no network, no disk, no clock beyond what a caller
passes in. Everything a provider adapter learns is funnelled into
:class:`EarningsEvidence`, and every number the Earnings Analyst report
displays is computed here. That split is what makes the interesting failure
modes testable without a vendor:

* a negative or zero-crossing EPS estimate,
* an analyst count that contradicts the revision counts,
* a period whose 7-day and 90-day trends point in opposite directions,
* a provider that discloses a trend but not the breadth behind it.

Three rules are load-bearing.

**A missing value is missing, never zero.** Every scalar travels as a
:class:`Value` carrying its own ``source``/``as_of`` and, when absent, a
``unavailable_reason`` string. ``0.0`` is a measurement; ``None`` is an
admission. Collapsing the two is how a stock with no analyst coverage starts
reporting "Neutral earnings momentum" — a claim nobody made.

**Percentage change is symmetric, not ordinary.** ``(new - old) / old`` is
wrong for this data in three separate ways: it divides by zero at ``old == 0``,
it *inverts sign* when ``old < 0`` (a loss narrowing from -2.60 to -2.44 reads
as a downgrade), and it explodes without bound near zero. Yahoo's own
``growth`` column has the sign bug — RIVN reports ``growth: 0.2383`` on a
negative EPS — which is why it is ignored here in favour of
:func:`symmetric_change`, bounded to ``[-2, 2]`` by construction.

**Momentum is a weighted mean of the signals that exist**, renormalized over
available weight, with an explicit floor below which the answer is
``Insufficient Data`` rather than a number. Silently treating an absent signal
as neutral drags every thinly-covered name toward the middle band and makes
"Neutral" mean two different things.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from tradingagents.dataflows.evidence_values import (
    EPSILON,
    Value,
    _fmt,
    _fmt_large,
    bounded,
    safe_date,
    safe_float,
    safe_int,
    safe_ratio,
)

# ---------------------------------------------------------------------------
# Locked constants
#
# The plan that introduced this module required these to be pinned by
# table-driven tests rather than tuned in place, because moving any of them
# silently changes the product meaning of a published "Strong Positive".
# tests/test_earnings_models.py asserts every boundary below.
#
# EPSILON, safe_float/safe_int/safe_date/safe_ratio, bounded, and
# Value/_fmt/_fmt_large now live in evidence_values.py — nothing earnings-
# specific about them, and Quality/Valuation reuse them rather than
# duplicating ~150 lines. Re-exported here (imported above) so nothing that
# already does ``from earnings_models import safe_float`` (etc.) breaks.
# ---------------------------------------------------------------------------

#: Symmetric change at which a horizon's signal reaches full strength (±1).
#: Scaled by horizon: a 2% move in seven days is as informative as an 8% move
#: in ninety, so the same raw change earns a smaller score at a longer horizon.
MOMENTUM_SCALES: dict[str, float] = {
    "eps_7d": 0.02,
    "eps_30d": 0.04,
    "eps_90d": 0.08,
    "revenue_30d": 0.03,
}

#: Signal weights. EPS direction dominates; breadth is a check on it; revenue
#: is a minority confirmation. Surprise, drift and guidance are deliberately
#: absent — they are context in the report and may not move the score.
MOMENTUM_WEIGHTS: dict[str, float] = {
    "eps_7d": 0.15,
    "eps_30d": 0.35,
    "eps_90d": 0.20,
    "breadth_30d": 0.20,
    "revenue_30d": 0.10,
}

#: Available weight must reach this before a band is published at all.
MIN_AVAILABLE_WEIGHT = 0.50

#: Band boundaries, applied to the renormalized score in ``[-1, 1]``.
BAND_STRONG_POSITIVE = 0.60
BAND_POSITIVE = 0.20
BAND_NEGATIVE = -0.20
BAND_STRONG_NEGATIVE = -0.60

#: Two EPS horizons count as conflicting only when both are this strong and
#: point opposite ways. Without a magnitude floor, numerical noise around zero
#: would flag a conflict on almost every symbol.
HORIZON_CONFLICT_FLOOR = 0.10

MomentumBand = Literal[
    "Strong Positive",
    "Positive",
    "Neutral",
    "Negative",
    "Strong Negative",
    "Insufficient Data",
]

Confidence = Literal["low", "medium", "high"]

_CONFIDENCE_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

#: Yahoo's relative period keys. ``0q``/``+1q`` are quarters, ``0y``/``+1y``
#: fiscal years. Kept as the wire keys so a provider payload is recognizable
#: in a saved snapshot years later.
PERIOD_CURRENT_QUARTER = "0q"
PERIOD_NEXT_QUARTER = "+1q"
PERIOD_CURRENT_YEAR = "0y"
PERIOD_NEXT_YEAR = "+1y"

_RELATIVE_PERIOD_LABELS = {
    PERIOD_CURRENT_QUARTER: "Current quarter (relative period 0q)",
    PERIOD_NEXT_QUARTER: "Next quarter (relative period +1q)",
    PERIOD_CURRENT_YEAR: "Current fiscal year (relative period 0y)",
    PERIOD_NEXT_YEAR: "Next fiscal year (relative period +1y)",
}

#: The period momentum is published against. The current fiscal year has the
#: broadest analyst coverage of the four, and is the period the requested
#: "FY consensus today vs 30d ago" presentation refers to.
PRIMARY_PERIOD = PERIOD_CURRENT_YEAR


# ---------------------------------------------------------------------------
# Change arithmetic
# ---------------------------------------------------------------------------


def symmetric_change(today: float | None, old: float | None) -> float | None:
    """Relative change that stays sane across zero and negative values.

    ``2 * (today - old) / (|today| + |old|)``, which is bounded to ``[-2, 2]``,
    keeps the sign of ``today - old`` regardless of either operand's sign, and
    is finite everywhere except a denominator of zero.

    Worked cases, all real payloads:

    * ``8.81249`` from ``8.76760`` (AAPL FY26) -> ``+0.0051``.
    * ``-2.43642`` from ``-2.60537`` (RIVN FY26) -> ``+0.0670``. A narrowing
      loss is an *upgrade*; ordinary percentage change reports ``-0.065``.
    * ``+0.10`` from ``-0.10`` (crossing zero) -> ``+2.0``, the maximum, rather
      than the ``-2.0`` ordinary change produces or the division by zero it
      produces at ``old == 0``.

    ``None`` propagates. Two values that are both indistinguishable from zero
    are ``0.0`` when equal and ``None`` when not: at that magnitude the
    direction is real but the ratio is meaningless, and reporting either
    ``0.0`` or ``±2.0`` would be a fabrication.
    """
    if today is None or old is None:
        return None
    denominator = abs(today) + abs(old)
    if denominator < EPSILON:
        return 0.0 if today == old else None
    return 2.0 * (today - old) / denominator


# ---------------------------------------------------------------------------
# Fiscal periods
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FiscalPeriod:
    """A provider's relative period key, resolved to a label when possible.

    ``0y`` alone cannot be printed as "FY27" — that would be inventing a fiscal
    calendar. When ``end_date`` is known the label carries it explicitly
    (``FY2027 (FYE 2027-01-31)``) rather than only the year, because a
    January-ending fiscal year is named inconsistently across issuers and a
    bare "FY2027" would be unverifiable. With no metadata the relative label is
    printed as-is.
    """

    key: str
    end_date: str | None = None

    @property
    def label(self) -> str:
        if self.end_date:
            year = self.end_date[:4]
            if self.key.endswith("q"):
                return f"Quarter ending {self.end_date}"
            return f"FY{year} (FYE {self.end_date})"
        return _RELATIVE_PERIOD_LABELS.get(self.key, f"Relative period {self.key}")

    @property
    def is_annual(self) -> bool:
        return self.key.endswith("y")

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "end_date": self.end_date, "label": self.label}

    @classmethod
    def from_dict(cls, raw: Any) -> FiscalPeriod:
        if not isinstance(raw, dict):
            return cls(key="unknown")
        return cls(key=str(raw.get("key") or "unknown"), end_date=raw.get("end_date"))


def resolve_annual_period_end(
    relative_key: str,
    *,
    next_fiscal_year_end: str | None,
) -> str | None:
    """Resolve ``0y``/``+1y`` to a fiscal-year-end date, or ``None``.

    ``next_fiscal_year_end`` is the end of the fiscal year currently in
    progress — which is the period Yahoo labels ``0y``, since ``0y`` is an
    *estimate* and an estimate only exists for a year that has not closed.
    ``+1y`` is that date plus one year. Nothing is guessed: with no metadata
    the caller keeps the relative label.
    """
    base = safe_date(next_fiscal_year_end)
    if base is None:
        return None
    if relative_key == PERIOD_CURRENT_YEAR:
        return base
    if relative_key == PERIOD_NEXT_YEAR:
        try:
            parsed = date.fromisoformat(base)
            return parsed.replace(year=parsed.year + 1).isoformat()
        except ValueError:
            # 29 Feb fiscal year end; step back a day rather than fail.
            parsed = date.fromisoformat(base)
            return date(parsed.year + 1, parsed.month, parsed.day - 1).isoformat()
    return None


# ---------------------------------------------------------------------------
# Per-period evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EstimateTrend:
    """Consensus now and at four lookbacks. Absent horizons stay absent."""

    current: Value = field(default_factory=lambda: Value.missing("not reported"))
    days_ago_7: Value = field(default_factory=lambda: Value.missing("not reported"))
    days_ago_30: Value = field(default_factory=lambda: Value.missing("not reported"))
    days_ago_60: Value = field(default_factory=lambda: Value.missing("not reported"))
    days_ago_90: Value = field(default_factory=lambda: Value.missing("not reported"))

    def change(self, horizon: str) -> float | None:
        """Symmetric change from ``horizon`` ("7d"/"30d"/"60d"/"90d") to now."""
        old = {
            "7d": self.days_ago_7,
            "30d": self.days_ago_30,
            "60d": self.days_ago_60,
            "90d": self.days_ago_90,
        }[horizon]
        return symmetric_change(self.current.value, old.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current.to_dict(),
            "days_ago_7": self.days_ago_7.to_dict(),
            "days_ago_30": self.days_ago_30.to_dict(),
            "days_ago_60": self.days_ago_60.to_dict(),
            "days_ago_90": self.days_ago_90.to_dict(),
            "change_7d": self.change("7d"),
            "change_30d": self.change("30d"),
            "change_60d": self.change("60d"),
            "change_90d": self.change("90d"),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> EstimateTrend:
        if not isinstance(raw, dict):
            return cls()
        return cls(
            current=Value.from_dict(raw.get("current")),
            days_ago_7=Value.from_dict(raw.get("days_ago_7")),
            days_ago_30=Value.from_dict(raw.get("days_ago_30")),
            days_ago_60=Value.from_dict(raw.get("days_ago_60")),
            days_ago_90=Value.from_dict(raw.get("days_ago_90")),
        )


@dataclass(frozen=True)
class RevisionBreadth:
    """Analyst up/down counts per window.

    90-day counts are modelled but expected to be unavailable: Yahoo publishes
    a 90-day *trend* and only 7/30-day *counts*. Reusing the 30-day counts to
    fill the 90-day row would be a fabrication that reads identically to a
    measurement, so the field stays missing and the report says so.
    """

    up_7d: Value = field(default_factory=lambda: Value.missing("not reported", unit="count"))
    down_7d: Value = field(default_factory=lambda: Value.missing("not reported", unit="count"))
    up_30d: Value = field(default_factory=lambda: Value.missing("not reported", unit="count"))
    down_30d: Value = field(default_factory=lambda: Value.missing("not reported", unit="count"))
    up_90d: Value = field(default_factory=lambda: Value.missing("not reported", unit="count"))
    down_90d: Value = field(default_factory=lambda: Value.missing("not reported", unit="count"))

    def net_ratio(self, window: str) -> float | None:
        """``(up - down) / (up + down)`` for ``"7d"``/``"30d"``/``"90d"``."""
        up, down = {
            "7d": (self.up_7d, self.down_7d),
            "30d": (self.up_30d, self.down_30d),
            "90d": (self.up_90d, self.down_90d),
        }[window]
        if not up.available or not down.available:
            return None
        return safe_ratio(up.value - down.value, up.value + down.value)

    def total(self, window: str) -> int | None:
        up, down = {
            "7d": (self.up_7d, self.down_7d),
            "30d": (self.up_30d, self.down_30d),
            "90d": (self.up_90d, self.down_90d),
        }[window]
        if not up.available or not down.available:
            return None
        return int(round(up.value + down.value))  # type: ignore[operator]

    def to_dict(self) -> dict[str, Any]:
        return {
            "up_7d": self.up_7d.to_dict(),
            "down_7d": self.down_7d.to_dict(),
            "up_30d": self.up_30d.to_dict(),
            "down_30d": self.down_30d.to_dict(),
            "up_90d": self.up_90d.to_dict(),
            "down_90d": self.down_90d.to_dict(),
            "net_ratio_7d": self.net_ratio("7d"),
            "net_ratio_30d": self.net_ratio("30d"),
            "net_ratio_90d": self.net_ratio("90d"),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> RevisionBreadth:
        if not isinstance(raw, dict):
            return cls()
        return cls(
            up_7d=Value.from_dict(raw.get("up_7d")),
            down_7d=Value.from_dict(raw.get("down_7d")),
            up_30d=Value.from_dict(raw.get("up_30d")),
            down_30d=Value.from_dict(raw.get("down_30d")),
            up_90d=Value.from_dict(raw.get("up_90d")),
            down_90d=Value.from_dict(raw.get("down_90d")),
        )


@dataclass(frozen=True)
class PeriodEvidence:
    """Everything known about one forecast period."""

    period: FiscalPeriod
    eps: EstimateTrend = field(default_factory=EstimateTrend)
    revenue: EstimateTrend = field(default_factory=EstimateTrend)
    breadth: RevisionBreadth = field(default_factory=RevisionBreadth)
    analyst_count: Value = field(
        default_factory=lambda: Value.missing("not reported", unit="count")
    )
    year_ago_eps: Value = field(default_factory=lambda: Value.missing("not reported"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period.to_dict(),
            "eps": self.eps.to_dict(),
            "revenue": self.revenue.to_dict(),
            "breadth": self.breadth.to_dict(),
            "analyst_count": self.analyst_count.to_dict(),
            "year_ago_eps": self.year_ago_eps.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> PeriodEvidence:
        if not isinstance(raw, dict):
            return cls(period=FiscalPeriod(key="unknown"))
        return cls(
            period=FiscalPeriod.from_dict(raw.get("period")),
            eps=EstimateTrend.from_dict(raw.get("eps")),
            revenue=EstimateTrend.from_dict(raw.get("revenue")),
            breadth=RevisionBreadth.from_dict(raw.get("breadth")),
            analyst_count=Value.from_dict(raw.get("analyst_count")),
            year_ago_eps=Value.from_dict(raw.get("year_ago_eps")),
        )


# ---------------------------------------------------------------------------
# Calendar, surprises, drift, guidance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EarningsCalendar:
    """The next announcement, and the uncertainty around it.

    ``date_is_estimated`` matters: Yahoo returns a range (two dates) when the
    issuer has not confirmed, and treating an unconfirmed window as a fixed
    date is how an earnings-blackout check silently passes.
    """

    next_date: str | None = None
    next_date_range_end: str | None = None
    date_is_estimated: bool = False
    timing: str | None = None  # "bmo" | "amc" | "unknown"
    eps_estimate_avg: Value = field(default_factory=lambda: Value.missing("not reported"))
    eps_estimate_low: Value = field(default_factory=lambda: Value.missing("not reported"))
    eps_estimate_high: Value = field(default_factory=lambda: Value.missing("not reported"))
    revenue_estimate_avg: Value = field(
        default_factory=lambda: Value.missing("not reported", unit="currency_large")
    )
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.next_date is not None

    def days_until(self, as_of: str) -> int | None:
        if self.next_date is None:
            return None
        try:
            return (date.fromisoformat(self.next_date) - date.fromisoformat(as_of)).days
        except ValueError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_date": self.next_date,
            "next_date_range_end": self.next_date_range_end,
            "date_is_estimated": self.date_is_estimated,
            "timing": self.timing,
            "eps_estimate_avg": self.eps_estimate_avg.to_dict(),
            "eps_estimate_low": self.eps_estimate_low.to_dict(),
            "eps_estimate_high": self.eps_estimate_high.to_dict(),
            "revenue_estimate_avg": self.revenue_estimate_avg.to_dict(),
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> EarningsCalendar:
        if not isinstance(raw, dict):
            return cls(unavailable_reason="malformed serialized calendar")
        return cls(
            next_date=raw.get("next_date"),
            next_date_range_end=raw.get("next_date_range_end"),
            date_is_estimated=bool(raw.get("date_is_estimated")),
            timing=raw.get("timing"),
            eps_estimate_avg=Value.from_dict(raw.get("eps_estimate_avg")),
            eps_estimate_low=Value.from_dict(raw.get("eps_estimate_low")),
            eps_estimate_high=Value.from_dict(raw.get("eps_estimate_high")),
            revenue_estimate_avg=Value.from_dict(raw.get("revenue_estimate_avg")),
            unavailable_reason=raw.get("unavailable_reason"),
        )


@dataclass(frozen=True)
class SurpriseEvent:
    """One reported quarter against its consensus.

    ``fiscal_period_end`` is what Yahoo's ``earnings_history`` is indexed by,
    and it is emphatically *not* the announcement date — the June quarter is
    reported in late July or August. ``announcement_date`` is therefore a
    separate, often-absent field, and the drift calculation refuses to run
    without it rather than anchoring to the quarter end and reporting a
    three-week-late window as the market reaction.
    """

    fiscal_period_end: str
    announcement_date: str | None = None
    eps_actual: Value = field(default_factory=lambda: Value.missing("not reported"))
    eps_estimate: Value = field(default_factory=lambda: Value.missing("not reported"))
    eps_difference: Value = field(default_factory=lambda: Value.missing("not reported"))
    surprise_pct: Value = field(
        default_factory=lambda: Value.missing("not reported", unit="pct_dec")
    )

    @property
    def beat(self) -> bool | None:
        if not self.eps_difference.available:
            return None
        return self.eps_difference.value > 0  # type: ignore[operator]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fiscal_period_end": self.fiscal_period_end,
            "announcement_date": self.announcement_date,
            "eps_actual": self.eps_actual.to_dict(),
            "eps_estimate": self.eps_estimate.to_dict(),
            "eps_difference": self.eps_difference.to_dict(),
            "surprise_pct": self.surprise_pct.to_dict(),
            "beat": self.beat,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> SurpriseEvent:
        if not isinstance(raw, dict):
            return cls(fiscal_period_end="unknown")
        return cls(
            fiscal_period_end=str(raw.get("fiscal_period_end") or "unknown"),
            announcement_date=raw.get("announcement_date"),
            eps_actual=Value.from_dict(raw.get("eps_actual")),
            eps_estimate=Value.from_dict(raw.get("eps_estimate")),
            eps_difference=Value.from_dict(raw.get("eps_difference")),
            surprise_pct=Value.from_dict(raw.get("surprise_pct")),
        )


@dataclass(frozen=True)
class DriftObservation:
    """Post-earnings drift over one horizon, in trading sessions.

    ``anchor_session`` records which session the window was measured from,
    because an announcement lands before the open, after the close, or on a
    holiday, and "the first tradable session at or after the announcement" is
    an assumption a reader must be able to check.
    """

    fiscal_period_end: str
    announcement_date: str
    anchor_session: str
    sessions: int
    stock_return: Value = field(
        default_factory=lambda: Value.missing("not computed", unit="pct_dec")
    )
    benchmark_return: Value = field(
        default_factory=lambda: Value.missing("not computed", unit="pct_dec")
    )
    excess_return: Value = field(
        default_factory=lambda: Value.missing("not computed", unit="pct_dec")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fiscal_period_end": self.fiscal_period_end,
            "announcement_date": self.announcement_date,
            "anchor_session": self.anchor_session,
            "sessions": self.sessions,
            "stock_return": self.stock_return.to_dict(),
            "benchmark_return": self.benchmark_return.to_dict(),
            "excess_return": self.excess_return.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> DriftObservation:
        if not isinstance(raw, dict):
            return cls(
                fiscal_period_end="unknown",
                announcement_date="unknown",
                anchor_session="unknown",
                sessions=0,
            )
        return cls(
            fiscal_period_end=str(raw.get("fiscal_period_end") or "unknown"),
            announcement_date=str(raw.get("announcement_date") or "unknown"),
            anchor_session=str(raw.get("anchor_session") or "unknown"),
            sessions=safe_int(raw.get("sessions")) or 0,
            stock_return=Value.from_dict(raw.get("stock_return")),
            benchmark_return=Value.from_dict(raw.get("benchmark_return")),
            excess_return=Value.from_dict(raw.get("excess_return")),
        )


@dataclass(frozen=True)
class GuidanceNote:
    """A sourced quote or summary. Never synthesized from price action."""

    text: str
    source: str
    published: str | None = None
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "published": self.published,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> GuidanceNote:
        if not isinstance(raw, dict):
            return cls(text="", source="unknown")
        return cls(
            text=str(raw.get("text") or ""),
            source=str(raw.get("source") or "unknown"),
            published=raw.get("published"),
            url=raw.get("url"),
        )


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MomentumAssessment:
    """The computed verdict, with the arithmetic left visible.

    ``signals``/``weights_used`` are published so a reader can reproduce
    ``score`` by hand. That is not decoration: a band is the one number in the
    report most likely to be quoted on its own, and an unreproducible band is
    indistinguishable from an LLM's guess.
    """

    band: MomentumBand
    score: float | None
    signals: dict[str, float] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    available_weight: float = 0.0
    confidence: Confidence = "low"
    period_key: str = PRIMARY_PERIOD
    discrepancies: list[str] = field(default_factory=list)
    missing_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "band": self.band,
            "score": self.score,
            "signals": dict(self.signals),
            "weights_used": dict(self.weights_used),
            "available_weight": self.available_weight,
            "confidence": self.confidence,
            "period_key": self.period_key,
            "discrepancies": list(self.discrepancies),
            "missing_signals": list(self.missing_signals),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> MomentumAssessment:
        if not isinstance(raw, dict):
            return cls(band="Insufficient Data", score=None)
        band = raw.get("band")
        if band not in {
            "Strong Positive", "Positive", "Neutral",
            "Negative", "Strong Negative", "Insufficient Data",
        }:
            band = "Insufficient Data"
        confidence = raw.get("confidence")
        if confidence not in _CONFIDENCE_ORDER:
            confidence = "low"
        return cls(
            band=band,  # type: ignore[arg-type]
            score=safe_float(raw.get("score")),
            signals={k: float(v) for k, v in (raw.get("signals") or {}).items()},
            weights_used={k: float(v) for k, v in (raw.get("weights_used") or {}).items()},
            available_weight=safe_float(raw.get("available_weight")) or 0.0,
            confidence=confidence,  # type: ignore[arg-type]
            period_key=str(raw.get("period_key") or PRIMARY_PERIOD),
            discrepancies=[str(x) for x in (raw.get("discrepancies") or [])],
            missing_signals=[str(x) for x in (raw.get("missing_signals") or [])],
        )


def period_sort_key(key: str) -> tuple[int, int, str]:
    """Chronological order for relative period keys.

    Plain string sort puts ``+1y`` before ``0y`` — ``+`` precedes ``0`` in ASCII —
    so a report table listed next year before this one. Quarters come before
    annual periods, each in ascending offset.
    """
    text = (key or "").strip()
    kind = 0 if text.endswith("q") else 1 if text.endswith("y") else 2
    body = text[:-1] if kind < 2 else text
    try:
        offset = int(body)
    except ValueError:
        return (kind, 999, text)
    return (kind, offset, text)


def band_for_score(score: float) -> MomentumBand:
    """Map a renormalized score in ``[-1, 1]`` onto a band.

    Boundaries are inclusive on the positive side and exclusive on the
    negative, so ``+0.20`` is Positive while ``-0.20`` is Negative. The
    asymmetry is deliberate and pinned by test: an ambiguous boundary is how
    the same score prints two different bands across refactors.
    """
    if score >= BAND_STRONG_POSITIVE:
        return "Strong Positive"
    if score >= BAND_POSITIVE:
        return "Positive"
    if score > BAND_NEGATIVE:
        return "Neutral"
    if score > BAND_STRONG_NEGATIVE:
        return "Negative"
    return "Strong Negative"


def _cap_confidence(current: Confidence, ceiling: Confidence) -> Confidence:
    if _CONFIDENCE_ORDER[ceiling] < _CONFIDENCE_ORDER[current]:
        return ceiling
    return current


def compute_momentum(period: PeriodEvidence) -> MomentumAssessment:
    """Score one period's revision activity and band it.

    Only signals that exist contribute, and the weights are renormalized over
    those present — an absent revenue trend must not pull the score toward zero
    as a phantom neutral vote. Below :data:`MIN_AVAILABLE_WEIGHT`, or with no
    EPS horizon at all, the answer is ``Insufficient Data``: a band computed
    from breadth alone would present an analyst headcount as an estimate
    direction.
    """
    raw_signals: dict[str, float | None] = {
        "eps_7d": bounded(period.eps.change("7d"), MOMENTUM_SCALES["eps_7d"]),
        "eps_30d": bounded(period.eps.change("30d"), MOMENTUM_SCALES["eps_30d"]),
        "eps_90d": bounded(period.eps.change("90d"), MOMENTUM_SCALES["eps_90d"]),
        "breadth_30d": period.breadth.net_ratio("30d"),
        "revenue_30d": bounded(
            period.revenue.change("30d"), MOMENTUM_SCALES["revenue_30d"]
        ),
    }

    signals = {name: value for name, value in raw_signals.items() if value is not None}
    missing = sorted(name for name, value in raw_signals.items() if value is None)
    weights_used = {name: MOMENTUM_WEIGHTS[name] for name in signals}
    available_weight = sum(weights_used.values())

    discrepancies: list[str] = []

    # Retain, never resolve: revision counts accumulate over a rolling window
    # while the analyst count is a point-in-time figure, so the counts legally
    # exceed it. Dropping either would hide a real provider inconsistency.
    total_30d = period.breadth.total("30d")
    analysts = safe_int(period.analyst_count.value)
    if total_30d is not None and analysts is not None and total_30d > analysts:
        discrepancies.append(
            f"30-day revision count ({total_30d}) exceeds reported analyst "
            f"coverage ({analysts}); counts are cumulative over the window while "
            "coverage is a point-in-time figure"
        )

    # A 30-day window contains its own 7-day window, so a 7-day count above the
    # 30-day one is internally impossible. Yahoo publishes it anyway — AAPL was
    # observed at 9 downgrades in 7 days against 8 in 30 — which means the two
    # columns are refreshed on different schedules. Retained rather than
    # smoothed: it is direct evidence about how much precision the breadth
    # signal actually carries.
    for direction, short, long_ in (
        ("upgrades", period.breadth.up_7d, period.breadth.up_30d),
        ("downgrades", period.breadth.down_7d, period.breadth.down_30d),
    ):
        if short.available and long_.available and short.value > long_.value:  # type: ignore[operator]
            discrepancies.append(
                f"7-day {direction} ({int(short.value)}) exceed 30-day "  # type: ignore[arg-type]
                f"({int(long_.value)}), which is arithmetically impossible; the "  # type: ignore[arg-type]
                "provider refreshes the two windows independently"
            )

    eps_signals = {k: v for k, v in signals.items() if k.startswith("eps_")}
    strong = [v for v in eps_signals.values() if abs(v) >= HORIZON_CONFLICT_FLOOR]
    horizons_conflict = any(a * b < 0 for a in strong for b in strong)
    if horizons_conflict:
        discrepancies.append(
            "EPS revision horizons disagree in direction: "
            + ", ".join(f"{k}={v:+.3f}" for k, v in sorted(eps_signals.items()))
        )

    if not eps_signals or available_weight < MIN_AVAILABLE_WEIGHT:
        return MomentumAssessment(
            band="Insufficient Data",
            score=None,
            signals=signals,
            weights_used=weights_used,
            available_weight=available_weight,
            confidence="low",
            period_key=period.period.key,
            discrepancies=discrepancies,
            missing_signals=missing,
        )

    score = sum(signals[name] * weights_used[name] for name in signals) / available_weight
    score = max(-1.0, min(1.0, score))

    confidence: Confidence = "high"
    if available_weight < 0.75:
        confidence = _cap_confidence(confidence, "medium")
    if analysts is None or analysts < 2:
        confidence = _cap_confidence(confidence, "low")
    elif analysts < 5:
        confidence = _cap_confidence(confidence, "medium")
    if horizons_conflict:
        confidence = _cap_confidence(confidence, "medium")

    return MomentumAssessment(
        band=band_for_score(score),
        score=score,
        signals=signals,
        weights_used=weights_used,
        available_weight=available_weight,
        confidence=confidence,
        period_key=period.period.key,
        discrepancies=discrepancies,
        missing_signals=missing,
    )


# ---------------------------------------------------------------------------
# The aggregate
# ---------------------------------------------------------------------------

EvidenceStatus = Literal["ok", "partial", "unsupported", "pit_unavailable", "no_coverage"]

#: Bumped whenever the serialized shape changes. Stored alongside every
#: snapshot so a future reader can tell a schema change from a data change.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EarningsEvidence:
    """One symbol's earnings evidence as of one date.

    ``status`` is the first thing a consumer should branch on. ``partial`` is
    the common case, not a failure: no free provider covers every field for
    every venue, and a partial answer with named gaps is more useful than a
    refusal.
    """

    symbol: str
    as_of: str
    status: EvidenceStatus = "ok"
    canonical_symbol: str | None = None
    company_name: str | None = None
    currency: str | None = None
    quote_type: str | None = None
    periods: dict[str, PeriodEvidence] = field(default_factory=dict)
    calendar: EarningsCalendar = field(default_factory=EarningsCalendar)
    surprises: list[SurpriseEvent] = field(default_factory=list)
    drift: list[DriftObservation] = field(default_factory=list)
    drift_unavailable_reason: str | None = None
    guidance: list[GuidanceNote] = field(default_factory=list)
    momentum: MomentumAssessment = field(
        default_factory=lambda: MomentumAssessment(band="Insufficient Data", score=None)
    )
    sources: list[str] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status_detail: str | None = None
    schema_version: int = SCHEMA_VERSION

    # -- factories for the non-ok paths ---------------------------------

    @classmethod
    def unsupported(cls, symbol: str, as_of: str, detail: str) -> EarningsEvidence:
        """An instrument that has no earnings at all (ETF, index, FX, crypto)."""
        return cls(
            symbol=symbol,
            as_of=as_of,
            status="unsupported",
            status_detail=detail,
            data_gaps=[detail],
        )

    @classmethod
    def pit_unavailable(cls, symbol: str, as_of: str, detail: str) -> EarningsEvidence:
        """A historical date with no snapshot observed on or before it.

        Deliberately not filled from a live endpoint. Today's consensus
        answering a question about a past date is a future function, and it is
        undetectable downstream: the numbers look exactly like a measurement.
        """
        return cls(
            symbol=symbol,
            as_of=as_of,
            status="pit_unavailable",
            status_detail=detail,
            data_gaps=[detail],
        )

    @classmethod
    def no_coverage(cls, symbol: str, as_of: str, detail: str) -> EarningsEvidence:
        """A company the provider covers for prices but not for estimates."""
        return cls(
            symbol=symbol,
            as_of=as_of,
            status="no_coverage",
            status_detail=detail,
            data_gaps=[detail],
        )

    # -- derived ---------------------------------------------------------

    @property
    def primary_period(self) -> PeriodEvidence | None:
        """The period momentum is published against, if present."""
        return self.periods.get(self.momentum.period_key) or self.periods.get(PRIMARY_PERIOD)

    def annual_periods(self) -> list[PeriodEvidence]:
        return [
            p for _, p in sorted(self.periods.items(), key=lambda kv: period_sort_key(kv[0]))
            if p.period.is_annual
        ]

    def quarterly_periods(self) -> list[PeriodEvidence]:
        return [
            p for _, p in sorted(self.periods.items(), key=lambda kv: period_sort_key(kv[0]))
            if not p.period.is_annual
        ]

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "canonical_symbol": self.canonical_symbol,
            "company_name": self.company_name,
            "currency": self.currency,
            "quote_type": self.quote_type,
            "as_of": self.as_of,
            "status": self.status,
            "status_detail": self.status_detail,
            "periods": {
                key: value.to_dict()
                for key, value in sorted(
                    self.periods.items(), key=lambda kv: period_sort_key(kv[0])
                )
            },
            "calendar": self.calendar.to_dict(),
            "surprises": [s.to_dict() for s in self.surprises],
            "drift": [d.to_dict() for d in self.drift],
            "drift_unavailable_reason": self.drift_unavailable_reason,
            "guidance": [g.to_dict() for g in self.guidance],
            "momentum": self.momentum.to_dict(),
            "sources": list(self.sources),
            "data_gaps": list(self.data_gaps),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> EarningsEvidence:
        if not isinstance(raw, dict):
            raise ValueError("earnings evidence payload is not an object")
        status = raw.get("status")
        if status not in {"ok", "partial", "unsupported", "pit_unavailable", "no_coverage"}:
            status = "partial"
        return cls(
            symbol=str(raw.get("symbol") or "unknown"),
            as_of=str(raw.get("as_of") or ""),
            status=status,  # type: ignore[arg-type]
            canonical_symbol=raw.get("canonical_symbol"),
            company_name=raw.get("company_name"),
            currency=raw.get("currency"),
            quote_type=raw.get("quote_type"),
            periods={
                str(key): PeriodEvidence.from_dict(value)
                for key, value in (raw.get("periods") or {}).items()
            },
            calendar=EarningsCalendar.from_dict(raw.get("calendar")),
            surprises=[SurpriseEvent.from_dict(s) for s in (raw.get("surprises") or [])],
            drift=[DriftObservation.from_dict(d) for d in (raw.get("drift") or [])],
            drift_unavailable_reason=raw.get("drift_unavailable_reason"),
            guidance=[GuidanceNote.from_dict(g) for g in (raw.get("guidance") or [])],
            momentum=MomentumAssessment.from_dict(raw.get("momentum")),
            sources=[str(s) for s in (raw.get("sources") or [])],
            data_gaps=[str(s) for s in (raw.get("data_gaps") or [])],
            warnings=[str(s) for s in (raw.get("warnings") or [])],
            status_detail=raw.get("status_detail"),
            schema_version=safe_int(raw.get("schema_version")) or SCHEMA_VERSION,
        )


def finalize_evidence(evidence: EarningsEvidence) -> EarningsEvidence:
    """Recompute momentum from the periods and settle ``status``.

    Called once by each adapter after it has populated periods, so the band is
    always derived from the evidence actually present rather than set by hand
    at a call site that may have been edited since.
    """
    from dataclasses import replace

    period = evidence.periods.get(PRIMARY_PERIOD)
    if period is None:
        annual = evidence.annual_periods()
        period = annual[0] if annual else None

    momentum = (
        compute_momentum(period)
        if period is not None
        else MomentumAssessment(band="Insufficient Data", score=None)
    )

    gaps = list(evidence.data_gaps)
    for name in momentum.missing_signals:
        gap = _MISSING_SIGNAL_GAPS.get(name)
        if gap and gap not in gaps:
            gaps.append(gap)

    status = evidence.status
    if status in {"ok", "partial"}:
        status = "ok" if momentum.band != "Insufficient Data" and not gaps else "partial"

    return replace(evidence, momentum=momentum, data_gaps=gaps, status=status)


_MISSING_SIGNAL_GAPS = {
    "eps_7d": "7-day EPS consensus trend unavailable",
    "eps_30d": "30-day EPS consensus trend unavailable",
    "eps_90d": "90-day EPS consensus trend unavailable",
    "breadth_30d": "30-day analyst revision breadth (up/down counts) unavailable",
    "revenue_30d": (
        "30-day revenue consensus trend unavailable; no free provider publishes a "
        "revenue revision history, so this needs two local point-in-time snapshots"
    ),
}


# ---------------------------------------------------------------------------
# Deterministic rendering
#
# Every number the report shows is produced here, from the evidence, with no
# language model in the path. The narrative sections are appended by the
# analyst; they cannot reach these values.
# ---------------------------------------------------------------------------


def render_evidence_report(evidence: EarningsEvidence) -> str:
    """Render the code-owned portion of the Earnings Analyst report."""
    if evidence.status in {"unsupported", "pit_unavailable", "no_coverage"}:
        return _render_terminal_status(evidence)

    parts = [
        _render_header(evidence),
        _render_momentum(evidence),
        _render_primary_consensus(evidence),
        _render_period_table(evidence),
        _render_calendar(evidence),
        _render_surprises(evidence),
        _render_drift(evidence),
        _render_sources(evidence),
    ]
    return "\n\n".join(p for p in parts if p)


def _render_terminal_status(evidence: EarningsEvidence) -> str:
    titles = {
        "unsupported": "Earnings analysis not applicable",
        "pit_unavailable": "Point-in-time earnings evidence unavailable",
        "no_coverage": "No analyst estimate coverage",
    }
    lines = [
        f"# Earnings & Estimate Revisions — {evidence.symbol}",
        "",
        f"**Status:** {titles[evidence.status]}",
        "",
        evidence.status_detail or "No detail supplied.",
        "",
        "No estimate, revision, surprise, or drift figures are reported for this "
        "request. Do not substitute values from another period, another symbol, "
        "or prior knowledge.",
    ]
    if evidence.sources:
        lines += ["", "**Sources consulted:** " + ", ".join(evidence.sources)]
    return "\n".join(lines)


def _render_header(evidence: EarningsEvidence) -> str:
    name = evidence.company_name or evidence.symbol
    lines = [
        f"# Earnings & Estimate Revisions — {name} ({evidence.symbol})",
        "",
        f"**As of:** {evidence.as_of}",
    ]
    if evidence.canonical_symbol and evidence.canonical_symbol != evidence.symbol:
        lines.append(f"**Queried as:** {evidence.canonical_symbol}")
    if evidence.currency:
        lines.append(f"**Reporting currency:** {evidence.currency}")
    if evidence.status == "partial":
        lines.append(
            "**Coverage:** partial — see Data Gaps. Figures shown are measured; "
            "absent fields are absent, not zero."
        )
    return "\n".join(lines)


def _render_momentum(evidence: EarningsEvidence) -> str:
    m = evidence.momentum
    period = evidence.periods.get(m.period_key)
    period_label = period.period.label if period else m.period_key
    lines = [
        "## Earnings Momentum",
        "",
        f"**{m.band}**"
        + (f"  (score {m.score:+.3f} on -1..+1)" if m.score is not None else ""),
        "",
        f"- Period scored: {period_label}",
        f"- Confidence: {m.confidence}",
        f"- Signal coverage: {m.available_weight:.2f} of 1.00 weight available",
    ]
    if m.signals:
        lines.append("")
        lines.append("| Signal | Value | Weight |")
        lines.append("| --- | ---: | ---: |")
        for name in sorted(m.signals):
            lines.append(
                f"| {name} | {m.signals[name]:+.3f} | {m.weights_used[name]:.2f} |"
            )
    if m.band == "Insufficient Data":
        lines += [
            "",
            "Momentum is not scored: the available revision signals do not meet "
            f"the {MIN_AVAILABLE_WEIGHT:.2f} weight floor, or no EPS trend horizon "
            "was published. This is a statement about data coverage, not a neutral "
            "verdict on the company.",
        ]
    if m.discrepancies:
        lines += ["", "**Retained discrepancies:**"]
        lines += [f"- {d}" for d in m.discrepancies]
    return "\n".join(lines)


def _render_primary_consensus(evidence: EarningsEvidence) -> str:
    """The requested headline: FY consensus today vs 30 days ago, plus breadth."""
    period = evidence.primary_period
    if period is None:
        return ""
    eps = period.eps
    lines = [
        f"## {period.period.label} EPS Consensus",
        "",
        "| Horizon | Consensus EPS |",
        "| --- | ---: |",
        f"| 90 days ago | {_fmt(eps.days_ago_90)} |",
        f"| 60 days ago | {_fmt(eps.days_ago_60)} |",
        f"| 30 days ago | {_fmt(eps.days_ago_30)} |",
        f"| 7 days ago | {_fmt(eps.days_ago_7)} |",
        f"| **Today** | **{_fmt(eps.current)}** |",
    ]

    changes = [
        ("7 days", eps.change("7d")),
        ("30 days", eps.change("30d")),
        ("90 days", eps.change("90d")),
    ]
    rendered = [
        f"- Over {label}: {value * 100:+.2f}% (symmetric)"
        for label, value in changes
        if value is not None
    ]
    if rendered:
        lines += ["", "**Estimate change**", *rendered]
        lines.append(
            "  Symmetric change is `2(new-old)/(|new|+|old|)`, which keeps its sign "
            "when EPS is negative or crosses zero. It is not an ordinary percentage."
        )

    lines += ["", "**Revision breadth**"]
    for window in ("7d", "30d", "90d"):
        up, down = {
            "7d": (period.breadth.up_7d, period.breadth.down_7d),
            "30d": (period.breadth.up_30d, period.breadth.down_30d),
            "90d": (period.breadth.up_90d, period.breadth.down_90d),
        }[window]
        if up.available and down.available:
            ratio = period.breadth.net_ratio(window)
            ratio_text = f", net {ratio:+.3f}" if ratio is not None else ""
            lines.append(
                f"- Last {window}: +{_fmt(up)} raised / -{_fmt(down)} lowered{ratio_text}"
            )
        else:
            reason = up.unavailable_reason or down.unavailable_reason or "not reported"
            lines.append(f"- Last {window}: unavailable ({reason})")

    lines.append(f"- Analysts covering this period: {_fmt(period.analyst_count)}")
    if period.revenue.current.available:
        lines += [
            "",
            "**Revenue consensus**",
            f"- Today: {_fmt(period.revenue.current)}",
        ]
        rev_30 = period.revenue.change("30d")
        if rev_30 is not None:
            lines.append(f"- Over 30 days: {rev_30 * 100:+.2f}% (symmetric)")
        else:
            lines.append(
                "- Over 30 days: unavailable — no consulted provider publishes a "
                "revenue revision history"
            )
    return "\n".join(lines)


def _render_period_table(evidence: EarningsEvidence) -> str:
    if not evidence.periods:
        return ""
    lines = [
        "## All Forecast Periods",
        "",
        "| Period | EPS today | 30d ago | 90d ago | Up 30d | Down 30d | Analysts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(evidence.periods, key=period_sort_key):
        p = evidence.periods[key]
        lines.append(
            f"| {p.period.label} | {_fmt(p.eps.current)} | {_fmt(p.eps.days_ago_30)} "
            f"| {_fmt(p.eps.days_ago_90)} | {_fmt(p.breadth.up_30d)} "
            f"| {_fmt(p.breadth.down_30d)} | {_fmt(p.analyst_count)} |"
        )
    return "\n".join(lines)


def _render_calendar(evidence: EarningsEvidence) -> str:
    cal = evidence.calendar
    lines = ["## Next Earnings Date", ""]
    if not cal.available:
        lines.append(
            f"Unavailable ({cal.unavailable_reason or 'not reported by any consulted provider'}). "
            "Do not infer a date from the reporting cadence."
        )
        return "\n".join(lines)

    when = cal.next_date
    if cal.next_date_range_end and cal.next_date_range_end != cal.next_date:
        when = f"{cal.next_date} — {cal.next_date_range_end}"
    days = cal.days_until(evidence.as_of)
    suffix = f" ({days:+d} days from as-of)" if days is not None else ""
    lines.append(f"- Date: {when}{suffix}")
    if cal.date_is_estimated:
        lines.append(
            "- ⚠️ Unconfirmed: the provider supplied a window, not an issuer-confirmed "
            "date. Any earnings-blackout rule must treat the whole window as in scope."
        )
    if cal.timing and cal.timing != "unknown":
        lines.append(f"- Timing: {cal.timing}")
    else:
        lines.append("- Timing (before/after market): unavailable")
    lines.append(f"- Consensus EPS: {_fmt(cal.eps_estimate_avg)}")
    if cal.eps_estimate_low.available or cal.eps_estimate_high.available:
        lines.append(
            f"- Consensus EPS range: {_fmt(cal.eps_estimate_low)} — {_fmt(cal.eps_estimate_high)}"
        )
    lines.append(f"- Consensus revenue: {_fmt(cal.revenue_estimate_avg)}")
    return "\n".join(lines)


def _render_surprises(evidence: EarningsEvidence) -> str:
    if not evidence.surprises:
        return (
            "## Surprise History\n\nUnavailable — no reported-quarter history was "
            "returned by any consulted provider."
        )
    lines = [
        "## Surprise History",
        "",
        "| Fiscal quarter end | Reported EPS | Consensus | Difference | Surprise | Announced |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in evidence.surprises:
        lines.append(
            f"| {s.fiscal_period_end} | {_fmt(s.eps_actual)} | {_fmt(s.eps_estimate)} "
            f"| {_fmt(s.eps_difference)} | {_fmt(s.surprise_pct)} "
            f"| {s.announcement_date or 'unavailable'} |"
        )
    beats = [s.beat for s in evidence.surprises if s.beat is not None]
    if beats:
        lines += [
            "",
            f"- Beat rate over the last {len(beats)} reported quarters: "
            f"{sum(1 for b in beats if b)}/{len(beats)}",
        ]
    lines.append(
        "- Quarter-end dates are fiscal period ends, not announcement dates. A "
        "restatement or provider correction can change a historical row, so these "
        "are the current vintage rather than the figures known at the time."
    )
    return "\n".join(lines)


def _render_drift(evidence: EarningsEvidence) -> str:
    lines = ["## Post-Earnings Drift", ""]
    if not evidence.drift:
        lines.append(
            "Unavailable — "
            + (
                evidence.drift_unavailable_reason
                or "announcement dates were not available, and drift cannot be anchored "
                "to a fiscal quarter end without misdating the market reaction."
            )
        )
        return "\n".join(lines)
    lines += [
        "| Quarter end | Announced | Anchor session | Sessions | Stock | Benchmark | Excess |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for d in evidence.drift:
        lines.append(
            f"| {d.fiscal_period_end} | {d.announcement_date} | {d.anchor_session} "
            f"| {d.sessions} | {_fmt(d.stock_return)} | {_fmt(d.benchmark_return)} "
            f"| {_fmt(d.excess_return)} |"
        )
    lines.append("")
    lines.append(
        "- Windows are measured in trading sessions from the first tradable session "
        "at or after the announcement, so a release on a holiday or after the close "
        "anchors to the next open session."
    )
    return "\n".join(lines)


def _render_sources(evidence: EarningsEvidence) -> str:
    lines = ["## Sources & Data Gaps", ""]
    lines.append(
        "**Sources:** " + (", ".join(evidence.sources) if evidence.sources else "none recorded")
    )
    lines.append("")
    if evidence.data_gaps:
        lines.append("**Data gaps (measured absences, not zeros):**")
        lines += [f"- {g}" for g in evidence.data_gaps]
    else:
        lines.append("**Data gaps:** none.")
    if evidence.warnings:
        lines += ["", "**Warnings:**"]
        lines += [f"- {w}" for w in evidence.warnings]
    return "\n".join(lines)

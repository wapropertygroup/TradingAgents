"""Normalized business-quality evidence, and the deterministic tier over it.

Mirrors ``earnings_models.py``'s shape: a normalized ``Evidence`` dataclass
built by an adapter from provider data, a pure ``compute_*`` function scoring
it against locked constants, and a pure ``render_*_report`` function — no
language model anywhere in this module. The Quality Analyst is only allowed to
narrate on top of what this file computes.

Why these particular signals
-----------------------------
The checklist comes from value/quality-investing frameworks (Buffett: durable
high ROE, competitive moat, low leverage; Munger: consistency of the whole
record, not one good year; Graham: financial strength via the current ratio)
translated into fields Yahoo's fundamentals actually publish. One signal is
not a simple point-in-time ratio: ``margin_consistency`` needs several periods
of operating margin, which is what actually distinguishes "quality" from a
single lucky quarter — Munger's own framing.

**These weights and scales are a first cut, not a calibrated model.** They are
locked constants pinned by ``tests/test_quality_models.py`` for the same
reason ``earnings_models.py``'s are: moving one silently changes what a
published "High Quality" means to a reader. Retuning them against realized
outcomes is future work (this repo has no backtesting harness yet); until
then they encode textbook thresholds and Graham's own literal numbers, cited
per signal below.

Unit note (verified against live yfinance data for AAPL/KO/NVDA)
------------------------------------------------------------------
``returnOnEquity``/``returnOnAssets``/``operatingMargins``/``profitMargins``
arrive as decimal fractions (0.15 = 15%) — used directly. ``debtToEquity``
arrives on yfinance's own 0-200-ish scale (AAPL 78.4, KO 115.5, NVDA 17.0),
which is *not* a decimal fraction of anything — the adapter divides by 100
before it reaches this module, so every ``Value`` here is a true ratio
(0.784x, not 78.4). Getting this wrong in either direction silently produces a
tier that is off by two orders of magnitude while still looking plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import pstdev
from typing import Any, Literal

from tradingagents.dataflows.evidence_values import (
    Value,
    _fmt,
    piecewise_score,
    safe_float,
    safe_int,
    safe_ratio,
    weighted_mean_score,
)

SCHEMA_VERSION = 1

EvidenceStatus = Literal["ok", "partial", "unsupported", "no_coverage"]

QualityTier = Literal[
    "High Quality", "Above Average", "Average", "Below Average", "Weak",
    "Insufficient Data",
]

#: Per-signal weights. ROE dominates (Buffett's #1 test); margin_consistency
#: is last and lightest because it is a derived, noisier statistic (it needs
#: several periods of income-statement data, any one of which may be absent).
QUALITY_WEIGHTS: dict[str, float] = {
    "roe": 0.25,
    "operating_margin": 0.20,
    "debt_to_equity": 0.15,
    "current_ratio": 0.15,
    "fcf_margin": 0.15,
    "margin_consistency": 0.10,
}

#: Available weight must reach this before a tier is published at all.
MIN_AVAILABLE_WEIGHT = 0.50

#: Band boundaries on the renormalized score in ``[-1, 1]``. Same asymmetric
#: (>= on the high side, > on the low side) convention as earnings_models.py's
#: band_for_score, so a boundary value prints one band consistently.
BAND_HIGH_QUALITY = 0.60
BAND_ABOVE_AVERAGE = 0.20
BAND_BELOW_AVERAGE = -0.20
BAND_WEAK = -0.60

#: Minimum number of income-statement periods before margin_consistency is
#: scored at all. Two points is not "consistency" -- it's a single change.
MIN_CONSISTENCY_PERIODS = 3
#: Above this operating-margin standard deviation (in decimal-fraction terms,
#: e.g. 0.08 = 8 points of margin), consistency scores -1. Below half of it,
#: +1. A textbook default, not a fitted one -- see module docstring.
CONSISTENCY_STDEV_FLOOR = 0.03
CONSISTENCY_STDEV_CEILING = 0.10


def _roe_score(v: float | None) -> float | None:
    return piecewise_score(v, low=0.0, mid=0.15, high=0.30)


def _operating_margin_score(v: float | None) -> float | None:
    return piecewise_score(v, low=0.0, mid=0.15, high=0.30)


def _debt_to_equity_score(v: float | None) -> float | None:
    # Decreasing triple: lower leverage scores higher. 0x -> +1, 1.0x -> 0,
    # 2.0x+ -> -1.
    return piecewise_score(v, low=2.0, mid=1.0, high=0.0)


def _current_ratio_score(v: float | None) -> float | None:
    # Graham's own literal financial-strength gate: "comfortably above 1.5".
    return piecewise_score(v, low=1.0, mid=1.5, high=2.5)


def _fcf_margin_score(v: float | None) -> float | None:
    return piecewise_score(v, low=0.0, mid=0.05, high=0.15)


def _margin_consistency_score(margin_history: list[float]) -> float | None:
    """Lower dispersion in operating margin across periods scores higher.

    Munger: "a great business earns high returns... year after year without
    heroic assumptions... consistency across the whole history, not one good
    year." This is the one signal in the tier that cannot be read off a single
    snapshot -- it is what actually operationalizes that sentence.
    """
    if len(margin_history) < MIN_CONSISTENCY_PERIODS:
        return None
    spread = pstdev(margin_history)
    # Decreasing triple: low spread is good.
    return piecewise_score(
        spread, low=CONSISTENCY_STDEV_CEILING,
        mid=(CONSISTENCY_STDEV_CEILING + CONSISTENCY_STDEV_FLOOR) / 2,
        high=CONSISTENCY_STDEV_FLOOR,
    )


def band_for_score(score: float) -> QualityTier:
    if score >= BAND_HIGH_QUALITY:
        return "High Quality"
    if score >= BAND_ABOVE_AVERAGE:
        return "Above Average"
    if score > BAND_BELOW_AVERAGE:
        return "Average"
    if score > BAND_WEAK:
        return "Below Average"
    return "Weak"


@dataclass(frozen=True)
class QualityTierAssessment:
    """The computed verdict, with the arithmetic left visible.

    ``signals``/``weights_used`` let a reader reproduce ``score`` by hand --
    the same reason earnings' ``MomentumAssessment`` publishes them: an
    unreproducible tier is indistinguishable from an LLM's guess.
    """

    tier: QualityTier
    score: float | None
    signals: dict[str, float] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    available_weight: float = 0.0
    missing_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier, "score": self.score,
            "signals": dict(self.signals), "weights_used": dict(self.weights_used),
            "available_weight": self.available_weight,
            "missing_signals": list(self.missing_signals),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> QualityTierAssessment:
        if not isinstance(raw, dict):
            return cls(tier="Insufficient Data", score=None)
        tier = raw.get("tier")
        if tier not in {"High Quality", "Above Average", "Average", "Below Average",
                        "Weak", "Insufficient Data"}:
            tier = "Insufficient Data"
        return cls(
            tier=tier, score=safe_float(raw.get("score")),
            signals={k: float(v) for k, v in (raw.get("signals") or {}).items()},
            weights_used={k: float(v) for k, v in (raw.get("weights_used") or {}).items()},
            available_weight=safe_float(raw.get("available_weight")) or 0.0,
            missing_signals=[str(x) for x in (raw.get("missing_signals") or [])],
        )


_MISSING_SIGNAL_GAPS = {
    "roe": "Return on equity unavailable",
    "operating_margin": "Operating margin unavailable",
    "debt_to_equity": "Debt-to-equity unavailable",
    "current_ratio": "Current ratio unavailable",
    "fcf_margin": "Free cash flow or revenue unavailable, so FCF margin could not be computed",
    "margin_consistency": (
        f"Fewer than {MIN_CONSISTENCY_PERIODS} periods of operating margin "
        "history available"
    ),
}


def compute_quality_tier(
    roe: float | None,
    operating_margin: float | None,
    debt_to_equity: float | None,
    current_ratio: float | None,
    fcf_margin: float | None,
    margin_history: list[float] | None = None,
) -> QualityTierAssessment:
    """Score one snapshot's business quality and band it.

    Every argument is a clean, already-unit-normalized ratio (see module
    docstring) -- this function does no vendor-format parsing.
    """
    raw_signals: dict[str, float | None] = {
        "roe": _roe_score(roe),
        "operating_margin": _operating_margin_score(operating_margin),
        "debt_to_equity": _debt_to_equity_score(debt_to_equity),
        "current_ratio": _current_ratio_score(current_ratio),
        "fcf_margin": _fcf_margin_score(fcf_margin),
        "margin_consistency": _margin_consistency_score(margin_history or []),
    }
    score, used, weights_used, available_weight, missing = weighted_mean_score(
        raw_signals, QUALITY_WEIGHTS, MIN_AVAILABLE_WEIGHT
    )
    if score is None:
        return QualityTierAssessment(
            tier="Insufficient Data", score=None, signals=used,
            weights_used=weights_used, available_weight=available_weight,
            missing_signals=missing,
        )
    return QualityTierAssessment(
        tier=band_for_score(score), score=score, signals=used,
        weights_used=weights_used, available_weight=available_weight,
        missing_signals=missing,
    )


@dataclass(frozen=True)
class QualityEvidence:
    """One symbol's business-quality evidence as of one date."""

    symbol: str
    as_of: str
    status: EvidenceStatus = "ok"
    company_name: str | None = None
    currency: str | None = None
    return_on_equity: Value = field(default_factory=lambda: Value.missing("not reported", unit="pct_dec"))
    operating_margin: Value = field(default_factory=lambda: Value.missing("not reported", unit="pct_dec"))
    profit_margin: Value = field(default_factory=lambda: Value.missing("not reported", unit="pct_dec"))
    return_on_assets: Value = field(default_factory=lambda: Value.missing("not reported", unit="pct_dec"))
    debt_to_equity: Value = field(default_factory=lambda: Value.missing("not reported", unit="ratio"))
    current_ratio: Value = field(default_factory=lambda: Value.missing("not reported", unit="ratio"))
    free_cash_flow: Value = field(default_factory=lambda: Value.missing("not reported", unit="currency_large"))
    total_revenue: Value = field(default_factory=lambda: Value.missing("not reported", unit="currency_large"))
    margin_history: list[Value] = field(default_factory=list)  # most-recent-first
    margin_history_periods: list[str] = field(default_factory=list)
    tier: QualityTierAssessment = field(
        default_factory=lambda: QualityTierAssessment(tier="Insufficient Data", score=None)
    )
    sources: list[str] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status_detail: str | None = None
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def unsupported(cls, symbol: str, as_of: str, detail: str) -> QualityEvidence:
        return cls(symbol=symbol, as_of=as_of, status="unsupported",
                   status_detail=detail, data_gaps=[detail])

    @classmethod
    def no_coverage(cls, symbol: str, as_of: str, detail: str) -> QualityEvidence:
        return cls(symbol=symbol, as_of=as_of, status="no_coverage",
                   status_detail=detail, data_gaps=[detail])

    @property
    def fcf_margin(self) -> float | None:
        return safe_ratio(self.free_cash_flow.value, self.total_revenue.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "symbol": self.symbol,
            "company_name": self.company_name, "currency": self.currency,
            "as_of": self.as_of, "status": self.status, "status_detail": self.status_detail,
            "return_on_equity": self.return_on_equity.to_dict(),
            "operating_margin": self.operating_margin.to_dict(),
            "profit_margin": self.profit_margin.to_dict(),
            "return_on_assets": self.return_on_assets.to_dict(),
            "debt_to_equity": self.debt_to_equity.to_dict(),
            "current_ratio": self.current_ratio.to_dict(),
            "free_cash_flow": self.free_cash_flow.to_dict(),
            "total_revenue": self.total_revenue.to_dict(),
            "margin_history": [v.to_dict() for v in self.margin_history],
            "margin_history_periods": list(self.margin_history_periods),
            "tier": self.tier.to_dict(),
            "sources": list(self.sources), "data_gaps": list(self.data_gaps),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> QualityEvidence:
        if not isinstance(raw, dict):
            raise ValueError("quality evidence payload is not an object")
        status = raw.get("status")
        if status not in {"ok", "partial", "unsupported", "no_coverage"}:
            status = "partial"
        return cls(
            symbol=str(raw.get("symbol") or "unknown"), as_of=str(raw.get("as_of") or ""),
            status=status, company_name=raw.get("company_name"), currency=raw.get("currency"),
            return_on_equity=Value.from_dict(raw.get("return_on_equity")),
            operating_margin=Value.from_dict(raw.get("operating_margin")),
            profit_margin=Value.from_dict(raw.get("profit_margin")),
            return_on_assets=Value.from_dict(raw.get("return_on_assets")),
            debt_to_equity=Value.from_dict(raw.get("debt_to_equity")),
            current_ratio=Value.from_dict(raw.get("current_ratio")),
            free_cash_flow=Value.from_dict(raw.get("free_cash_flow")),
            total_revenue=Value.from_dict(raw.get("total_revenue")),
            margin_history=[Value.from_dict(v) for v in (raw.get("margin_history") or [])],
            margin_history_periods=[str(p) for p in (raw.get("margin_history_periods") or [])],
            tier=QualityTierAssessment.from_dict(raw.get("tier")),
            sources=[str(s) for s in (raw.get("sources") or [])],
            data_gaps=[str(s) for s in (raw.get("data_gaps") or [])],
            warnings=[str(s) for s in (raw.get("warnings") or [])],
            status_detail=raw.get("status_detail"),
            schema_version=safe_int(raw.get("schema_version")) or SCHEMA_VERSION,
        )


def finalize_evidence(evidence: QualityEvidence) -> QualityEvidence:
    """Recompute the tier from the evidence and settle ``status``."""
    from dataclasses import replace

    history = [v.value for v in evidence.margin_history if v.available]
    tier = compute_quality_tier(
        roe=evidence.return_on_equity.value,
        operating_margin=evidence.operating_margin.value,
        debt_to_equity=evidence.debt_to_equity.value,
        current_ratio=evidence.current_ratio.value,
        fcf_margin=evidence.fcf_margin,
        margin_history=history,
    )
    gaps = list(evidence.data_gaps)
    for name in tier.missing_signals:
        gap = _MISSING_SIGNAL_GAPS.get(name)
        if gap and gap not in gaps:
            gaps.append(gap)
    status = evidence.status
    if status in {"ok", "partial"}:
        status = "ok" if tier.tier != "Insufficient Data" and not gaps else "partial"
    return replace(evidence, tier=tier, data_gaps=gaps, status=status)


def render_quality_report(evidence: QualityEvidence) -> str:
    """Render the code-owned portion of the Quality Analyst report."""
    if evidence.status in {"unsupported", "no_coverage"}:
        return _render_terminal_status(evidence)

    e = evidence
    name = e.company_name or e.symbol
    lines = [
        f"# Business Quality — {name} ({e.symbol})",
        "",
        f"**As of:** {e.as_of}",
    ]
    if e.status == "partial":
        lines.append(
            "**Coverage:** partial — see Data Gaps. Figures shown are measured; "
            "absent fields are absent, not zero."
        )
    lines += ["", "## Quality Tier", "", f"**{e.tier.tier}**"
             + (f"  (score {e.tier.score:+.3f} on -1..+1)" if e.tier.score is not None else "")]
    lines += [
        "",
        f"- Signal coverage: {e.tier.available_weight:.2f} of 1.00 weight available",
    ]
    if e.tier.signals:
        lines += ["", "| Signal | Value | Weight |", "| --- | ---: | ---: |"]
        for sig in sorted(e.tier.signals):
            lines.append(f"| {sig} | {e.tier.signals[sig]:+.3f} | {e.tier.weights_used[sig]:.2f} |")
    if e.tier.tier == "Insufficient Data":
        lines += [
            "",
            "Quality is not scored: the available signals do not meet the "
            f"{MIN_AVAILABLE_WEIGHT:.2f} weight floor. This is a statement about "
            "data coverage, not a neutral verdict on the business.",
        ]

    lines += [
        "",
        "## Profitability & Balance Sheet",
        "",
        f"- Return on equity: {_fmt(e.return_on_equity)}",
        f"- Operating margin: {_fmt(e.operating_margin)}",
        f"- Profit margin: {_fmt(e.profit_margin)}",
        f"- Return on assets: {_fmt(e.return_on_assets)}",
        f"- Debt to equity: {_fmt(e.debt_to_equity)}",
        f"- Current ratio: {_fmt(e.current_ratio)}",
        f"- Free cash flow: {_fmt(e.free_cash_flow)}",
    ]
    fcf_margin = e.fcf_margin
    if fcf_margin is not None:
        lines.append(f"- Free cash flow margin: {fcf_margin * 100:+.2f}%")

    if e.margin_history:
        lines += ["", "## Operating Margin History", "",
                 "| Period | Operating Margin |", "| --- | ---: |"]
        for period, val in zip(e.margin_history_periods, e.margin_history):
            lines.append(f"| {period} | {_fmt(val)} |")

    lines += ["", "## Sources & Data Gaps", "",
             "**Sources:** " + (", ".join(e.sources) if e.sources else "none recorded")]
    if e.data_gaps:
        lines += ["", "**Data gaps (measured absences, not zeros):**"]
        lines += [f"- {g}" for g in e.data_gaps]
    else:
        lines += ["", "**Data gaps:** none."]
    if e.warnings:
        lines += ["", "**Warnings:**"] + [f"- {w}" for w in e.warnings]
    return "\n".join(lines)


def _render_terminal_status(evidence: QualityEvidence) -> str:
    titles = {"unsupported": "Quality analysis not applicable",
             "no_coverage": "No fundamentals coverage"}
    lines = [
        f"# Business Quality — {evidence.symbol}", "",
        f"**Status:** {titles[evidence.status]}", "",
        evidence.status_detail or "No detail supplied.", "",
        "No quality figures are reported for this request. Do not substitute "
        "values from another symbol or prior knowledge.",
    ]
    if evidence.sources:
        lines += ["", "**Sources consulted:** " + ", ".join(evidence.sources)]
    return "\n".join(lines)

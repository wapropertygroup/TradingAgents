"""Normalized valuation evidence, and the deterministic tier over it.

Same shape as ``quality_models.py`` / ``earnings_models.py``: a normalized
``Evidence`` dataclass built by an adapter, a pure ``compute_*`` function
scoring it against locked constants, a pure ``render_*_report`` function. No
language model anywhere in this module.

Why these particular signals
-----------------------------
Graham's own literal numbers for the P/E band ("far above 15-20 demands
extraordinary justification"), Lynch's PEG test (growth cheap relative to its
own P/E), Graham's second lens (price-to-book), a consistency check between
two already-published numbers (forward vs. trailing P/E — is the market
already pricing in a change in earnings), and a minor Graham-era income tilt
(dividend yield, weighted low so a zero-dividend grower is not punished for
what it is).

**Locked constants, not a calibrated model** — same caveat as
``quality_models.py``: a first cut pinned by
``tests/test_valuation_models.py``, not tuned against realized outcomes.

**Negative or undefined P/E is missing, not a real signal value.** A company
with negative trailing EPS has no meaningful trailing P/E; treating a vendor's
negative or absent figure as an extreme "cheap" or "expensive" score would
invert the direction of a real trap (an unprofitable business does not become
attractive by definition). The adapter (``fundamentals_evidence.py``) is
responsible for turning that case into an absent ``Value`` before it reaches
this module — see its module docstring.

Sector-relative valuation was deliberately not attempted: no general-market
peer-comparison data source exists anywhere in this fork today
(``get_industry_comparison`` is A-share-only). Every band here is absolute,
the same way Graham's own checklist is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from tradingagents.dataflows.evidence_values import (
    Value,
    _fmt,
    bounded,
    piecewise_score,
    safe_float,
    safe_int,
    safe_ratio,
    weighted_mean_score,
)

SCHEMA_VERSION = 1

EvidenceStatus = Literal["ok", "partial", "unsupported", "no_coverage"]

ValuationTier = Literal[
    "Deep Value", "Attractive", "Fair", "Expensive", "Extreme Premium",
    "Insufficient Data",
]

#: P/E carries the most weight because it is Graham's own primary test and the
#: figure a reader already has intuition for; PEG next because it is the one
#: signal that adjusts for growth rather than taking the multiple at face
#: value. Dividend yield is lightest and never penalizes a zero, so a
#: zero-dividend grower is not marked down for a policy choice.
VALUATION_WEIGHTS: dict[str, float] = {
    "pe_band": 0.30,
    "peg": 0.25,
    "price_to_book": 0.20,
    "forward_vs_trailing": 0.15,
    "dividend_yield": 0.10,
}

MIN_AVAILABLE_WEIGHT = 0.50

BAND_DEEP_VALUE = 0.60
BAND_ATTRACTIVE = 0.20
BAND_EXPENSIVE = -0.20
BAND_EXTREME_PREMIUM = -0.60


def _pe_band_score(v: float | None) -> float | None:
    # Decreasing triple: cheaper P/E scores higher. 30x -> -1 ("far above
    # 15-20 demands extraordinary justification"), 17.5x (midpoint of
    # Graham's own 15-20 range) -> 0, 10x -> +1.
    return piecewise_score(v, low=30.0, mid=17.5, high=10.0)


def _peg_score(v: float | None) -> float | None:
    # Lynch's PEG test: below 1 is attractive, above 2 is not.
    return piecewise_score(v, low=2.5, mid=1.5, high=0.5)


def _price_to_book_score(v: float | None) -> float | None:
    # Graham's second lens.
    return piecewise_score(v, low=5.0, mid=3.0, high=1.5)


def _forward_vs_trailing_score(forward_pe: float | None, trailing_pe: float | None) -> float | None:
    """Positive when the market is already pricing in improving earnings.

    A forward P/E meaningfully below trailing means consensus expects EPS to
    grow into the multiple; the reverse means consensus expects it to shrink.
    This is a consistency check between two numbers the vendor already
    publishes, not a new field.
    """
    ratio = safe_ratio(
        (forward_pe - trailing_pe) if (forward_pe is not None and trailing_pe is not None) else None,
        trailing_pe,
    )
    if ratio is None:
        return None
    return bounded(-ratio, scale=0.30)


def _dividend_yield_score(pct: float | None) -> float | None:
    """0% is neutral, not penalized; a 5%+ yield saturates at +1."""
    return bounded(pct, scale=0.05)


def band_for_score(score: float) -> ValuationTier:
    if score >= BAND_DEEP_VALUE:
        return "Deep Value"
    if score >= BAND_ATTRACTIVE:
        return "Attractive"
    if score > BAND_EXPENSIVE:
        return "Fair"
    if score > BAND_EXTREME_PREMIUM:
        return "Expensive"
    return "Extreme Premium"


@dataclass(frozen=True)
class ValuationTierAssessment:
    tier: ValuationTier
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
    def from_dict(cls, raw: Any) -> ValuationTierAssessment:
        if not isinstance(raw, dict):
            return cls(tier="Insufficient Data", score=None)
        tier = raw.get("tier")
        if tier not in {"Deep Value", "Attractive", "Fair", "Expensive",
                        "Extreme Premium", "Insufficient Data"}:
            tier = "Insufficient Data"
        return cls(
            tier=tier, score=safe_float(raw.get("score")),
            signals={k: float(v) for k, v in (raw.get("signals") or {}).items()},
            weights_used={k: float(v) for k, v in (raw.get("weights_used") or {}).items()},
            available_weight=safe_float(raw.get("available_weight")) or 0.0,
            missing_signals=[str(x) for x in (raw.get("missing_signals") or [])],
        )


_MISSING_SIGNAL_GAPS = {
    "pe_band": "Trailing P/E unavailable or undefined (commonly a negative-earnings company)",
    "peg": "PEG ratio unavailable",
    "price_to_book": "Price-to-book unavailable",
    "forward_vs_trailing": "Forward P/E or trailing P/E unavailable, so the comparison could not be computed",
    "dividend_yield": "Dividend yield unavailable",
}


def compute_valuation_tier(
    trailing_pe: float | None,
    forward_pe: float | None,
    peg_ratio: float | None,
    price_to_book: float | None,
    dividend_yield_pct: float | None,
) -> ValuationTierAssessment:
    """Score one snapshot's valuation and band it.

    Every argument is already unit-normalized (see module docstring):
    ``dividend_yield_pct`` is a decimal fraction (0.0238 = 2.38%), not
    yfinance's raw percentage-point number.
    """
    raw_signals: dict[str, float | None] = {
        "pe_band": _pe_band_score(trailing_pe),
        "peg": _peg_score(peg_ratio),
        "price_to_book": _price_to_book_score(price_to_book),
        "forward_vs_trailing": _forward_vs_trailing_score(forward_pe, trailing_pe),
        "dividend_yield": _dividend_yield_score(dividend_yield_pct),
    }
    score, used, weights_used, available_weight, missing = weighted_mean_score(
        raw_signals, VALUATION_WEIGHTS, MIN_AVAILABLE_WEIGHT
    )
    if score is None:
        return ValuationTierAssessment(
            tier="Insufficient Data", score=None, signals=used,
            weights_used=weights_used, available_weight=available_weight,
            missing_signals=missing,
        )
    return ValuationTierAssessment(
        tier=band_for_score(score), score=score, signals=used,
        weights_used=weights_used, available_weight=available_weight,
        missing_signals=missing,
    )


@dataclass(frozen=True)
class ValuationEvidence:
    """One symbol's valuation evidence as of one date."""

    symbol: str
    as_of: str
    status: EvidenceStatus = "ok"
    company_name: str | None = None
    currency: str | None = None
    trailing_pe: Value = field(default_factory=lambda: Value.missing("not reported", unit="ratio"))
    forward_pe: Value = field(default_factory=lambda: Value.missing("not reported", unit="ratio"))
    peg_ratio: Value = field(default_factory=lambda: Value.missing("not reported", unit="ratio"))
    price_to_book: Value = field(default_factory=lambda: Value.missing("not reported", unit="ratio"))
    dividend_yield: Value = field(default_factory=lambda: Value.missing("not reported", unit="pct_dec"))
    market_cap: Value = field(default_factory=lambda: Value.missing("not reported", unit="currency_large"))
    tier: ValuationTierAssessment = field(
        default_factory=lambda: ValuationTierAssessment(tier="Insufficient Data", score=None)
    )
    sources: list[str] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status_detail: str | None = None
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def unsupported(cls, symbol: str, as_of: str, detail: str) -> ValuationEvidence:
        return cls(symbol=symbol, as_of=as_of, status="unsupported",
                   status_detail=detail, data_gaps=[detail])

    @classmethod
    def no_coverage(cls, symbol: str, as_of: str, detail: str) -> ValuationEvidence:
        return cls(symbol=symbol, as_of=as_of, status="no_coverage",
                   status_detail=detail, data_gaps=[detail])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "symbol": self.symbol,
            "company_name": self.company_name, "currency": self.currency,
            "as_of": self.as_of, "status": self.status, "status_detail": self.status_detail,
            "trailing_pe": self.trailing_pe.to_dict(), "forward_pe": self.forward_pe.to_dict(),
            "peg_ratio": self.peg_ratio.to_dict(), "price_to_book": self.price_to_book.to_dict(),
            "dividend_yield": self.dividend_yield.to_dict(), "market_cap": self.market_cap.to_dict(),
            "tier": self.tier.to_dict(),
            "sources": list(self.sources), "data_gaps": list(self.data_gaps),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> ValuationEvidence:
        if not isinstance(raw, dict):
            raise ValueError("valuation evidence payload is not an object")
        status = raw.get("status")
        if status not in {"ok", "partial", "unsupported", "no_coverage"}:
            status = "partial"
        return cls(
            symbol=str(raw.get("symbol") or "unknown"), as_of=str(raw.get("as_of") or ""),
            status=status, company_name=raw.get("company_name"), currency=raw.get("currency"),
            trailing_pe=Value.from_dict(raw.get("trailing_pe")),
            forward_pe=Value.from_dict(raw.get("forward_pe")),
            peg_ratio=Value.from_dict(raw.get("peg_ratio")),
            price_to_book=Value.from_dict(raw.get("price_to_book")),
            dividend_yield=Value.from_dict(raw.get("dividend_yield")),
            market_cap=Value.from_dict(raw.get("market_cap")),
            tier=ValuationTierAssessment.from_dict(raw.get("tier")),
            sources=[str(s) for s in (raw.get("sources") or [])],
            data_gaps=[str(s) for s in (raw.get("data_gaps") or [])],
            warnings=[str(s) for s in (raw.get("warnings") or [])],
            status_detail=raw.get("status_detail"),
            schema_version=safe_int(raw.get("schema_version")) or SCHEMA_VERSION,
        )


def finalize_evidence(evidence: ValuationEvidence) -> ValuationEvidence:
    """Recompute the tier from the evidence and settle ``status``."""
    from dataclasses import replace

    tier = compute_valuation_tier(
        trailing_pe=evidence.trailing_pe.value,
        forward_pe=evidence.forward_pe.value,
        peg_ratio=evidence.peg_ratio.value,
        price_to_book=evidence.price_to_book.value,
        dividend_yield_pct=evidence.dividend_yield.value,
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


def render_valuation_report(evidence: ValuationEvidence) -> str:
    """Render the code-owned portion of the Valuation Analyst report."""
    if evidence.status in {"unsupported", "no_coverage"}:
        return _render_terminal_status(evidence)

    e = evidence
    name = e.company_name or e.symbol
    lines = [
        f"# Valuation — {name} ({e.symbol})",
        "",
        f"**As of:** {e.as_of}",
    ]
    if e.status == "partial":
        lines.append(
            "**Coverage:** partial — see Data Gaps. Figures shown are measured; "
            "absent fields are absent, not zero."
        )
    lines += ["", "## Valuation Tier", "", f"**{e.tier.tier}**"
             + (f"  (score {e.tier.score:+.3f} on -1..+1)" if e.tier.score is not None else "")]
    lines += ["", f"- Signal coverage: {e.tier.available_weight:.2f} of 1.00 weight available"]
    if e.tier.signals:
        lines += ["", "| Signal | Value | Weight |", "| --- | ---: | ---: |"]
        for sig in sorted(e.tier.signals):
            lines.append(f"| {sig} | {e.tier.signals[sig]:+.3f} | {e.tier.weights_used[sig]:.2f} |")
    if e.tier.tier == "Insufficient Data":
        lines += [
            "",
            "Valuation is not scored: the available signals do not meet the "
            f"{MIN_AVAILABLE_WEIGHT:.2f} weight floor. This is a statement about "
            "data coverage (commonly a negative-earnings company with no "
            "trailing P/E), not a neutral verdict on price.",
        ]

    lines += [
        "",
        "## Multiples",
        "",
        f"- Trailing P/E: {_fmt(e.trailing_pe)}",
        f"- Forward P/E: {_fmt(e.forward_pe)}",
        f"- PEG ratio: {_fmt(e.peg_ratio)}",
        f"- Price to book: {_fmt(e.price_to_book)}",
        f"- Dividend yield: {_fmt(e.dividend_yield)}",
        f"- Market cap: {_fmt(e.market_cap)}",
    ]

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


def _render_terminal_status(evidence: ValuationEvidence) -> str:
    titles = {"unsupported": "Valuation analysis not applicable",
             "no_coverage": "No valuation coverage"}
    lines = [
        f"# Valuation — {evidence.symbol}", "",
        f"**Status:** {titles[evidence.status]}", "",
        evidence.status_detail or "No detail supplied.", "",
        "No valuation figures are reported for this request. Do not substitute "
        "values from another symbol or prior knowledge.",
    ]
    if evidence.sources:
        lines += ["", "**Sources consulted:** " + ", ".join(evidence.sources)]
    return "\n".join(lines)

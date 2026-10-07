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

Two enterprise-value multiples joined on 2026-10-06. EV/EBIT is an earnings
multiple that does not care how the company is financed: Greenblatt's earnings
yield read the other way up, 8x being a 12.5% yield. EV/Sales is the one
multiple a company with no profit still has, which is why it is weighted lower
(a sales multiple says nothing about margins) and why it matters: a
loss-maker's P/E is missing, and before it the tier rested on PEG and
price-to-book alone. The adapter builds both on one currency basis and leaves
them out for a bank or an insurer, whose debt is raw material rather than
financing.

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
    bounded,
    piecewise_score,
    render_sources_section,
    safe_float,
    safe_int,
    safe_ratio,
    weighted_mean_score,
)
from tradingagents.report_language import ReportText, for_language

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
#: EV/EBIT sits beside P/E as a second earnings multiple, and EV/Sales below it.
VALUATION_WEIGHTS: dict[str, float] = {
    "pe_band": 0.20,
    "peg": 0.20,
    "price_to_book": 0.15,
    "ev_ebit": 0.15,
    "ev_sales": 0.10,
    "forward_vs_trailing": 0.10,
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


def _ev_ebit_score(v: float | None) -> float | None:
    # 8x (a 12.5% earnings yield on the whole enterprise) -> +1, 15x -> 0,
    # 25x (a 4% yield) -> -1.
    return piecewise_score(v, low=25.0, mid=15.0, high=8.0)


def _ev_sales_score(v: float | None) -> float | None:
    # 1x -> +1, 3x (about the broad US market's) -> 0, 8x -> -1. Absolute, as
    # every band here is: a software company reads expensive on it by design.
    return piecewise_score(v, low=8.0, mid=3.0, high=1.0)


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
    "ev_sales": "EV/Sales unavailable",
    "ev_ebit": "EV/EBIT unavailable or undefined (an operating loss, or a bank or insurer)",
}


def compute_valuation_tier(
    trailing_pe: float | None,
    forward_pe: float | None,
    peg_ratio: float | None,
    price_to_book: float | None,
    dividend_yield_pct: float | None,
    ev_to_sales: float | None = None,
    ev_to_ebit: float | None = None,
) -> ValuationTierAssessment:
    """Score one snapshot's valuation and band it.

    Every argument is already unit-normalized (see module docstring):
    ``dividend_yield_pct`` is a decimal fraction (0.0238 = 2.38%), not
    yfinance's raw percentage-point number. The two EV multiples default to
    absent, which renormalizes onto the other signals.
    """
    raw_signals: dict[str, float | None] = {
        "pe_band": _pe_band_score(trailing_pe),
        "peg": _peg_score(peg_ratio),
        "price_to_book": _price_to_book_score(price_to_book),
        "ev_ebit": _ev_ebit_score(ev_to_ebit),
        "ev_sales": _ev_sales_score(ev_to_sales),
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
    enterprise_value: Value = field(
        default_factory=lambda: Value.missing("not reported", unit="currency_large"))
    ev_to_sales: Value = field(default_factory=lambda: Value.missing("not reported", unit="ratio"))
    ev_to_ebit: Value = field(default_factory=lambda: Value.missing("not reported", unit="ratio"))
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
            "enterprise_value": self.enterprise_value.to_dict(),
            "ev_to_sales": self.ev_to_sales.to_dict(), "ev_to_ebit": self.ev_to_ebit.to_dict(),
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
            # Absent from a payload written before 2026-10-06, which reads back
            # as missing rather than failing.
            enterprise_value=Value.from_dict(raw["enterprise_value"]) if "enterprise_value" in raw
            else Value.missing("not reported", unit="currency_large"),
            ev_to_sales=Value.from_dict(raw["ev_to_sales"]) if "ev_to_sales" in raw
            else Value.missing("not reported", unit="ratio"),
            ev_to_ebit=Value.from_dict(raw["ev_to_ebit"]) if "ev_to_ebit" in raw
            else Value.missing("not reported", unit="ratio"),
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
        ev_to_sales=evidence.ev_to_sales.value,
        ev_to_ebit=evidence.ev_to_ebit.value,
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


def render_valuation_report(evidence: ValuationEvidence, language: str | None = None) -> str:
    """Render the code-owned portion of the Valuation Analyst report.

    ``language`` is an ``output_language`` value, ``None`` for the run's own.
    """
    L = for_language(language)
    if evidence.status in {"unsupported", "no_coverage"}:
        return _render_terminal_status(evidence, L)

    e = evidence
    name = e.company_name or e.symbol
    lines = [
        L(f"# Valuation — {name} ({e.symbol})", f"# 估值 — {name}（{e.symbol}）"),
        "",
        L(f"**As of:** {e.as_of}", f"**截至：** {e.as_of}"),
    ]
    if e.status == "partial":
        lines.append(L(
            "**Coverage:** partial — see Data Gaps. Figures shown are measured; "
            "absent fields are absent, not zero.",
            "**覆盖：** 部分——见“数据缺口”。所列数字均为实测值；缺失的字段就是缺失，"
            "并非零值。",
        ))
    score = e.tier.score
    lines += ["", L("## Valuation Tier", "## 估值评级"), "", f"**{L.term(e.tier.tier)}**"
             + (L(f"  (score {score:+.3f} on -1..+1)", f"（评分 {score:+.3f}，区间 -1 至 +1）")
                if score is not None else "")]
    lines += ["", L(f"- Signal coverage: {e.tier.available_weight:.2f} of 1.00 weight available",
                    f"- 信号覆盖：可用权重 {e.tier.available_weight:.2f} / 1.00")]
    if e.tier.signals:
        lines += ["", L("| Signal | Value | Weight |", "| 信号 | 数值 | 权重 |"), "| --- | ---: | ---: |"]
        for sig in sorted(e.tier.signals):
            lines.append(f"| {L.signal(sig)} | {e.tier.signals[sig]:+.3f} "
                         f"| {e.tier.weights_used[sig]:.2f} |")
    if e.tier.tier == "Insufficient Data":
        lines += [
            "",
            L("Valuation is not scored: the available signals do not meet the "
              f"{MIN_AVAILABLE_WEIGHT:.2f} weight floor. This is a statement about "
              "data coverage (commonly a negative-earnings company with no "
              "trailing P/E), not a neutral verdict on price.",
              f"估值未评分：可用信号未达到 {MIN_AVAILABLE_WEIGHT:.2f} 的权重下限。"
              "这是对数据覆盖的说明（常见于没有历史市盈率的亏损公司），而非对价格的中性判断。"),
        ]

    lines += [
        "",
        L("## Multiples", "## 估值倍数"),
        "",
        L(f"- Trailing P/E: {L.fmt(e.trailing_pe)}", f"- 历史市盈率：{L.fmt(e.trailing_pe)}"),
        L(f"- Forward P/E: {L.fmt(e.forward_pe)}", f"- 远期市盈率：{L.fmt(e.forward_pe)}"),
        L(f"- PEG ratio: {L.fmt(e.peg_ratio)}", f"- PEG 比率：{L.fmt(e.peg_ratio)}"),
        L(f"- Price to book: {L.fmt(e.price_to_book)}", f"- 市净率：{L.fmt(e.price_to_book)}"),
        L(f"- Dividend yield: {L.fmt(e.dividend_yield)}", f"- 股息率：{L.fmt(e.dividend_yield)}"),
        L(f"- EV/Sales: {L.fmt(e.ev_to_sales)}", f"- 企业价值/营收：{L.fmt(e.ev_to_sales)}"),
        L(f"- EV/EBIT: {L.fmt(e.ev_to_ebit)}", f"- 企业价值/EBIT：{L.fmt(e.ev_to_ebit)}"),
        L(f"- Market cap: {L.fmt(e.market_cap)}", f"- 市值：{L.fmt(e.market_cap)}"),
        L(f"- Enterprise value: {L.fmt(e.enterprise_value)}",
          f"- 企业价值：{L.fmt(e.enterprise_value)}"),
    ]

    lines += ["", render_sources_section(e.sources, e.data_gaps, e.warnings, L)]
    return "\n".join(lines)


def _render_terminal_status(evidence: ValuationEvidence, L: ReportText) -> str:
    titles = {"unsupported": L("Valuation analysis not applicable", "估值分析不适用"),
             "no_coverage": L("No valuation coverage", "无估值数据覆盖")}
    lines = [
        L(f"# Valuation — {evidence.symbol}", f"# 估值 — {evidence.symbol}"), "",
        L(f"**Status:** {titles[evidence.status]}", f"**状态：** {titles[evidence.status]}"), "",
        L.message(evidence.status_detail or "No detail supplied."), "",
        L("No valuation figures are reported for this request. Do not substitute "
          "values from another symbol or prior knowledge.",
          "本次请求不报告任何估值数据。请勿用其他代码或既有知识中的数值替代。"),
    ]
    if evidence.sources:
        sources = L.join(L.source(s) for s in evidence.sources)
        lines += ["", L(f"**Sources consulted:** {sources}", f"**已查询的数据源：** {sources}")]
    return "\n".join(lines)

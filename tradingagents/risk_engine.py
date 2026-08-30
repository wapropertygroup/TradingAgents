"""
tradingagents.risk_engine
~~~~~~~~~~~~~~~~~~~~~~~~~
The deterministic gate between the Trader's proposal and the Portfolio Manager.

Why a gate at all
-----------------
Everything upstream of here is a language model. The debate produces a size and a
set of levels, and nothing in the pipeline checks them against the holder's stated
limits — a report can recommend 8% of a portfolio that already holds 14% of the
same company through two funds, and read perfectly well. This module is the one
place in the run where an answer is arithmetic rather than argument, and its
verdict is **binding**: the prompts downstream say so, because a constraint a
model may talk its way past is not a constraint.

Why the band arithmetic is not here
-----------------------------------
Deciding whether a size breaches a look-through limit needs the floor/ceiling
band, which needs a fund-composition cache and a recursive walk — neither of which
exists in this process. Re-implementing it here would be a second copy of
arithmetic that must agree exactly with the first, and the two would drift.

So the caller does it. ystocker's ``exposure.size_ladder`` evaluates its own
policy at a grid of candidate sizes and ships the resulting **ladder**; this module
looks a proposal up in it. All band arithmetic stays in one implementation and this
file does comparisons.

A ladder rather than a formula, because the feasible sizes are not an interval
anchored at zero. Both directions were measured on real fixtures: a portfolio of
100% AAPL against an 8% limit breaches at every small size and *starts passing*
above a large purchase of a bond fund, because the purchase dilutes the breach;
and a bond portfolio buying QQQ passes up to a point and breaches above it, because
MSFT is 32% of that fund.

The two rules this implements
----------------------------
**A breach clamps, it does not reject.** A run costs about fifteen minutes and real
API spend, so throwing the analysis away over a size is wasteful when the same
analysis at a smaller size is exactly what the holder wants. The gate returns the
largest tested size that is allowed and says it reduced it.

**Clamping is downward only.** A larger trade can legitimately pass where a smaller
one breaches — see the dilution case above — but a gate that responded by
*enlarging* a position would be inventing a trade nobody proposed, on the reasoning
that a bigger bet fixes a risk limit. Rungs above the proposal are ignored. The
ladder still carries them so the fact can be shown to a human.

**Missing or inconsistent levels block.** Not clamp: block. If the Trader's
structured output fell back to free text there are no numbers at all, and if the
numbers are internally inconsistent — a stop above the entry on a Buy, a size of
"0.05" that could mean either five percent or one hundredth of it — then there is
nothing trustworthy to clamp *to*. Sizing down an unverifiable proposal would
produce a smaller position justified by the same unchecked figures.

Absence is not zero, throughout
-------------------------------
No portfolio attached is :data:`GATE_NOT_EVALUATED`, not a pass and not a block:
most callers of this framework have no holdings, and blocking them all would make
the gate a bug rather than a control. An unstated cash balance is likewise not a
zero balance. Every one of those distinctions has already been got wrong once
somewhere in this pipeline, which is why they are named individually below.

Pure: stdlib only, no state, no I/O. Every input arrives as a plain mapping.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

#: The proposal is allowed exactly as stated.
GATE_PASS = "pass"
#: Allowed, but at a smaller size than proposed. ``approved_size_pct`` is binding.
GATE_CLAMPED = "clamped"
#: Do not place this trade.
GATE_BLOCKED = "blocked"
#: There was nothing to check against — no portfolio was supplied.
GATE_NOT_EVALUATED = "not_evaluated"

#: Verdict strings as ``exposure`` emits them. Only ``pass`` is a pass; an
#: unverifiable limit is not a satisfied one.
_OK_VERDICTS = frozenset({"pass"})

#: Reasons, in words. Coded keys travel in the payload and are turned into text
#: here so a report and a log line cannot describe the same refusal differently.
REASON_TEXT: dict[str, str] = {
    "no_structured_levels": (
        "the Trader's proposal came back as prose rather than structured fields, "
        "so it contains no numbers to check"),
    "incomplete_levels": (
        "the proposal is missing at least one of entry, stop, target or size, so "
        "it cannot be checked against the holder's limits"),
    "inconsistent_levels": (
        "the proposal's own levels contradict each other, so there is nothing "
        "trustworthy to size down to"),
    "no_allowed_size": (
        "no tested size at or below the proposal keeps the holder's stated limits, "
        "so this trade should not be placed at all"),
    "clamped_to_limit": (
        "the proposed size would breach a stated limit; it has been reduced to the "
        "largest tested size that does not"),
    "clamped_to_cash": (
        "the proposed size exceeds the stated cash balance and has been reduced to "
        "what the cash covers"),
    "already_breached": (
        "the portfolio already breaches this limit before the trade, so the "
        "breach is not caused by this proposal and cannot be fixed by resizing it"),
    "no_size_proposed": (
        "the proposal states no position size, so there is nothing to check "
        "against a size limit"),
}


def evaluate(trader_levels: Optional[Mapping[str, Any]],
             portfolio: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Gate a Trader proposal against a portfolio ladder. Pure arithmetic.

    ``trader_levels`` is ``TraderProposal.levels()`` — ``{}`` or None on the
    free-text path, which means the numbers do not exist.

    ``portfolio`` is ``exposure.size_ladder``'s output, or None when the caller
    attached no holdings.

    Returns a decision mapping. ``approved_size_pct`` is the size the Portfolio
    Manager must not exceed; ``None`` means either "no size was proposed" or "do
    not place this", which :data:`verdict` disambiguates.
    """
    levels = dict(trader_levels or {})
    action = str(levels.get("action") or "").strip().lower()
    proposed = levels.get("position_size_pct")
    reasons: list[str] = []

    # No portfolio: nothing to check against. Not a pass — the limits were never
    # evaluated — and not a block, since most callers have no holdings at all.
    # A ladder with no rungs counts as no portfolio: it can check nothing, and
    # reporting that as a block would refuse a trade on the strength of an empty
    # table.
    if not portfolio or not _rows(portfolio):
        return _decision(GATE_NOT_EVALUATED, None, proposed, [], levels, portfolio)

    # A Hold is not a transaction. Gating it would refuse to leave a position
    # alone, which is the one action that never needs permission.
    if action == "hold":
        return _decision(GATE_PASS, None, proposed, [], levels, portfolio)

    if not levels:
        return _decision(GATE_BLOCKED, None, None,
                         ["no_structured_levels"], levels, portfolio)

    # Inconsistent before incomplete: a contradiction is the more specific fault
    # and the more useful thing to report.
    if levels.get("flags"):
        reasons.append("inconsistent_levels")
        return _decision(GATE_BLOCKED, None, proposed, reasons, levels, portfolio)

    if not levels.get("complete"):
        return _decision(GATE_BLOCKED, None, proposed,
                         ["incomplete_levels"], levels, portfolio)

    if proposed is None:
        # complete() should have caught this; kept because a caller may hand in a
        # hand-built mapping and a missing size must not read as zero.
        return _decision(GATE_BLOCKED, None, None,
                         ["no_size_proposed"], levels, portfolio)

    allowed = _largest_allowed(portfolio, float(proposed))
    if allowed is None:
        reasons.append("no_allowed_size")
        if _baseline_breached(portfolio):
            # Worth separating: the holder is already over the limit, so no size of
            # this trade fixes it and the finding is about the portfolio.
            reasons.append("already_breached")
        return _decision(GATE_BLOCKED, None, proposed, reasons, levels, portfolio)

    approved = allowed
    cash_cap = _cash_cap_pct(portfolio)
    if cash_cap is not None and cash_cap < approved:
        approved = cash_cap
        reasons.append("clamped_to_cash")
    if allowed < float(proposed) - 1e-9:
        reasons.append("clamped_to_limit")

    if approved <= 1e-9:
        # "Clamped to 0%" is a refusal wearing the word approved. A zero-size
        # position is not a trade, and a Portfolio Manager reading CLAMPED will
        # describe it as permission granted at a smaller size.
        reasons.append("no_allowed_size")
        return _decision(GATE_BLOCKED, None, proposed, reasons, levels, portfolio)

    verdict = GATE_PASS if approved >= float(proposed) - 1e-9 else GATE_CLAMPED
    return _decision(verdict, approved, proposed, reasons, levels, portfolio)


def _rows(portfolio: Optional[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Ladder rows that are actually mappings.

    Tolerant on purpose. This runs inside a graph that has already spent every
    analyst call, so a malformed ladder must degrade to "nothing to check" rather
    than raise -- and a plain string in ``rungs`` iterates into characters, whose
    ``.get`` does not exist.
    """
    rungs = (portfolio or {}).get("rungs")
    if not isinstance(rungs, (list, tuple)):
        return []
    return [r for r in rungs if isinstance(r, Mapping)]


def _largest_allowed(portfolio: Mapping[str, Any], proposed_pct: float,
                     ) -> Optional[float]:
    """Largest ladder rung at or below *proposed_pct* whose verdict is acceptable.

    Downward only, deliberately: see the module docstring. ``None`` when no rung at
    or below the proposal is acceptable — which the caller must treat as "do not
    place this", never as zero.
    """
    best: Optional[float] = None
    for row in _rows(portfolio):
        try:
            pct = float(row.get("pct"))
        except (TypeError, ValueError):
            continue
        if pct <= proposed_pct + 1e-9 and row.get("verdict") in _OK_VERDICTS:
            if best is None or pct > best:
                best = pct
    return best


def _baseline_breached(portfolio: Mapping[str, Any]) -> bool:
    """Whether the portfolio breaches a limit at the zero rung, before any trade."""
    for row in _rows(portfolio):
        try:
            if abs(float(row.get("pct"))) < 1e-9:
                return row.get("verdict") == "breach"
        except (TypeError, ValueError):
            continue
    return False


def _cash_cap_pct(portfolio: Mapping[str, Any]) -> Optional[float]:
    """The largest size the stated cash covers, as a percent of portfolio value.

    ``None`` when no balance was stated — an unknown balance cannot cap anything,
    and treating silence as zero would refuse every buy. That exact bug shipped
    once on the ystocker side, where ``cash`` defaulted to 0.0.
    """
    cash = portfolio.get("cash")
    total = portfolio.get("total_value")
    if cash is None or not total:
        return None
    try:
        return max(0.0, float(cash) / float(total) * 100.0)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _decision(verdict: str, approved: Optional[float],
              proposed: Optional[float], reasons: list[str],
              levels: Mapping[str, Any],
              portfolio: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    ladder = _rows(portfolio)
    return {
        "verdict": verdict,
        "approved_size_pct": approved,
        "proposed_size_pct": proposed,
        "reasons": list(reasons),
        "binding": verdict in (GATE_CLAMPED, GATE_BLOCKED),
        "symbol": (portfolio or {}).get("symbol") or "",
        "reward_risk": levels.get("reward_risk"),
        "level_flags": list(levels.get("flags") or []),
        # Stated so a consumer can say what was not tested instead of implying the
        # grid was exhaustive. A silent cap reads as full coverage.
        "rungs_tested": len(ladder),
        "max_rung_pct": max(_pcts(ladder), default=0.0),
        "passing_rungs_above_proposal": _passing_above(ladder, proposed),
    }


def _pcts(ladder: Any) -> list[float]:
    """Every parseable rung percentage. Unparseable rows are skipped, not fatal."""
    out = []
    for row in ladder or []:
        if not isinstance(row, Mapping):
            continue
        try:
            out.append(float(row.get("pct")))
        except (TypeError, ValueError):
            continue
    return out


def _passing_above(ladder: Any, proposed: Optional[float]) -> list[float]:
    """Rungs larger than the proposal that would pass. Reported, never acted on.

    Exists so a human can see that dilution would satisfy the limit. The gate must
    not act on it: enlarging a position to clear a concentration check is not a
    risk control, and no automated step should reach that conclusion on its own.
    """
    if proposed is None:
        return []
    out = []
    for row in ladder or []:
        if not isinstance(row, Mapping):
            continue
        try:
            pct = float(row.get("pct"))
        except (TypeError, ValueError):
            continue
        if pct > float(proposed) + 1e-9 and row.get("verdict") in _OK_VERDICTS:
            out.append(pct)
    return sorted(out)


def render(decision: Mapping[str, Any]) -> str:
    """The decision as a text block for the prompts downstream.

    Framed as a ruling rather than as advice. The Portfolio Manager is a language
    model reading a report full of persuasive argument, and a constraint phrased as
    a suggestion is one it will weigh against the argument rather than obey.
    """
    verdict = str(decision.get("verdict") or GATE_NOT_EVALUATED)
    lines = ["<start_of_risk_gate>"]

    if verdict == GATE_NOT_EVALUATED:
        lines.extend([
            "RISK GATE: not evaluated — no portfolio was supplied with this run.",
            "This is not an approval. No position limit has been checked, so do "
            "not describe the trade as within any limit.",
            "<end_of_risk_gate>"])
        return "\n".join(lines)

    symbol = decision.get("symbol") or "the instrument"
    proposed = decision.get("proposed_size_pct")
    approved = decision.get("approved_size_pct")

    lines.append(f"RISK GATE RULING on {symbol}: {verdict.upper()}")
    lines.append(
        "This ruling was computed by a deterministic engine from the holder's "
        "stated limits and their look-through exposures. It is BINDING. Do not "
        "recalculate it, do not argue against it, and do not recommend a size "
        "above the approved one, however strong the case in the analysis.")
    lines.append("")

    if proposed is not None:
        lines.append(f"Size proposed by the Trader: {proposed}% of portfolio value")
    if verdict == GATE_BLOCKED:
        lines.append("Approved size: NONE — this trade must not be placed at the "
                     "proposed size.")
        lines.append("A rating may still be Hold or a reduction; it must not be an "
                     "instruction to buy at the proposed size.")
    elif approved is not None:
        lines.append(f"Approved size: {approved}% of portfolio value — this is a "
                     f"ceiling, not a target.")

    reasons = [REASON_TEXT.get(r, r) for r in (decision.get("reasons") or [])]
    if reasons:
        lines.append("")
        lines.append("Why:")
        lines.extend(f"  - {r}" for r in reasons)

    if decision.get("reward_risk") is not None:
        lines.append("")
        lines.append(f"Reward-to-risk of the proposal as stated: "
                     f"{decision['reward_risk']}:1 (computed, not quoted)")

    above = decision.get("passing_rungs_above_proposal") or []
    if above:
        lines.append("")
        lines.append(
            f"For information only: larger sizes ({', '.join(f'{p}%' for p in above[:4])}"
            f") would satisfy the limits, because a larger purchase dilutes an "
            f"existing concentration. The gate does not act on this and neither "
            f"should you — increasing a position to clear a risk limit is not risk "
            f"management. Raise it with the holder instead.")

    lines.append("")
    lines.append(f"Sizes tested: {decision.get('rungs_tested', 0)} rungs up to "
                 f"{decision.get('max_rung_pct', 0)}% of portfolio value. Sizes "
                 f"between rungs were not tested and are not implicitly approved.")
    lines.append("<end_of_risk_gate>")
    return "\n".join(lines)

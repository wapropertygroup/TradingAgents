"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    if isinstance(value, str) and value.strip().lower() in _NULLISH_FLOAT:
        return None
    return value


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: float | None = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    target_price: float | None = Field(
        default=None,
        description=(
            "Optional take-profit / price target in the instrument's quote "
            "currency. Give this whenever a stop-loss is given: without it the "
            "reward-to-risk ratio of the trade cannot be computed at all."
        ),
    )
    position_size_pct: float | None = Field(
        default=None,
        description=(
            "Optional position size as a PERCENTAGE of total portfolio value, "
            "expressed 0-100 — write 5 for five percent, not 0.05. Omit it "
            "entirely rather than guessing; an omitted size is treated as "
            "unstated, whereas a wrong one is acted on."
        ),
    )
    position_sizing: str | None = Field(
        default=None,
        description=(
            "Optional sizing guidance in words, e.g. 'scale in over three "
            "tranches'. Nuance that a single percentage cannot carry. Put the "
            "number in position_size_pct, not here."
        ),
    )

    @field_validator("entry_price", "stop_loss", "target_price",
                     "position_size_pct", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    # NOTE: there is deliberately no cross-field validator here, and that is a
    # correctness decision rather than an omission. ``invoke_structured_or_freetext``
    # catches *any* structured failure and retries once as free text, so a
    # validator that rejected an inconsistent proposal would not surface the
    # inconsistency — it would discard the whole structured object and every
    # number in it, leaving a downstream engine with nothing to check. Cross-field
    # facts are therefore computed, not enforced: see :meth:`levels`.

    def levels(self) -> dict[str, object]:
        """The numeric facts of this proposal, plus what they imply. Pure Python.

        This exists so a deterministic risk engine has structured numbers to
        check. Everything a language model produced is passed through unchanged;
        everything derived is arithmetic done here, because a reward-to-risk ratio
        computed by the model is a plausible-looking number nobody can audit.

        ``None`` means *unstated* throughout and never zero. A consumer must not
        read a missing stop as "no risk" or a missing size as "no position".

        ``flags`` names what is wrong rather than raising, and is empty when
        nothing is. Each flag is a fact about the proposal, not an opinion:

        ``size_ambiguous``
            A size of 0 < pct <= 1. The field is documented as 0-100, but "0.05"
            is a very plausible way for a model to write five percent, and the two
            readings differ by 100x. Neither is assumed — silently rescaling would
            risk a position a hundred times the intended one, and silently
            accepting risks one a hundredth the size. The engine treats the size
            as unverified.
        ``size_out_of_range``
            A size above 100%, which is not a percentage of a portfolio.
        ``stop_not_below_entry`` / ``stop_not_above_entry``
            A stop on the wrong side of the entry for the stated direction. For a
            Buy the stop belongs below; for a Sell, above.
        ``target_wrong_side``
            A target that does not profit in the stated direction.
        ``levels_without_direction``
            Prices given on a Hold, where "entry" and "stop" have no side and so
            no side can be checked.
        """
        action = self.action.value.lower()
        entry, stop = self.entry_price, self.stop_loss
        target, size = self.target_price, self.position_size_pct
        flags: list[str] = []

        if size is not None:
            if 0 < size <= 1:
                flags.append("size_ambiguous")
            elif size > 100:
                flags.append("size_out_of_range")

        if entry is not None and stop is not None:
            if action == "buy" and stop >= entry:
                flags.append("stop_not_below_entry")
            elif action == "sell" and stop <= entry:
                flags.append("stop_not_above_entry")
        if entry is not None and target is not None:
            if action == "buy" and target <= entry:
                flags.append("target_wrong_side")
            elif action == "sell" and target >= entry:
                flags.append("target_wrong_side")
        if action == "hold" and any(v is not None for v in (entry, stop, target)):
            flags.append("levels_without_direction")

        # Reward-to-risk, only when all three levels exist and both legs are
        # positive in the stated direction. A negative or zero risk leg means the
        # stop is on the wrong side, already flagged above, and dividing by it
        # would emit a number that reads as a real ratio.
        reward_risk = None
        if entry is not None and stop is not None and target is not None:
            reward = target - entry if action == "buy" else entry - target
            risk = entry - stop if action == "buy" else stop - entry
            if risk > 0 and reward > 0:
                reward_risk = round(reward / risk, 4)

        return {
            "action": self.action.value,
            "entry_price": entry,
            "stop_loss": stop,
            "target_price": target,
            "position_size_pct": size,
            "position_sizing_text": self.position_sizing or None,
            "reward_risk": reward_risk,
            # True only when every level needed for a risk check is present. The
            # engine reports an unchecked proposal rather than passing it.
            "complete": entry is not None and stop is not None
                        and target is not None and size is not None,
            "flags": flags,
        }


#: Flags from ``TraderProposal.levels()`` in words, for the rendered markdown.
#: Phrased as observations rather than instructions: the Portfolio Manager decides
#: what to do about an inconsistent proposal, and a renderer that told it to
#: discount the trade would be making that call from the wrong place.
_FLAG_TEXT = {
    "size_ambiguous": ("position size is ambiguous — written as a value at or "
                       "below 1, which could mean either that many percent or "
                       "one hundredth of it; treat the size as unstated"),
    "size_out_of_range": "position size exceeds 100% of the portfolio",
    "stop_not_below_entry": "stop-loss is not below the entry, on a Buy",
    "stop_not_above_entry": "stop-loss is not above the entry, on a Sell",
    "target_wrong_side": "price target does not profit in the stated direction",
    "levels_without_direction": ("entry/stop/target given on a Hold, where they "
                                 "have no side"),
    "size_zero_on_a_buy": ("the rating is a Buy but the position size is 0% — the "
                           "rating reads as an instruction and the size cancels it"),
}


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.target_price is not None:
        parts.extend(["", f"**Target Price**: {proposal.target_price}"])
    if proposal.position_size_pct is not None:
        parts.extend(["", f"**Position Size**: {proposal.position_size_pct}% of portfolio"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    # Derived, not asked for. The ratio is arithmetic over the three levels above
    # and is rendered so the reader and the Portfolio Manager see the same figure a
    # risk engine will check, rather than each computing their own.
    levels = proposal.levels()
    if levels["reward_risk"] is not None:
        parts.extend(["", f"**Reward:Risk**: {levels['reward_risk']}:1"])
    # Said out loud, because the alternative is worse than silence. An
    # inconsistent set of levels renders as perfectly plausible numbers -- a stop
    # above the entry on a Buy looks like a price -- and the ratio simply goes
    # missing, so a reader downstream has nothing to notice. Naming the problem is
    # what makes it reviewable.
    if levels["flags"]:
        parts.extend(["", "**Level Warnings**: " + ", ".join(
            _FLAG_TEXT.get(f, f) for f in levels["flags"])])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    position_size_pct: float | None = Field(
        default=None,
        description=(
            "The final recommended position size as a PERCENTAGE of total "
            "portfolio value, expressed 0-100 — write 5 for five percent, not "
            "0.05. If a risk gate ruling in the prompt approved a maximum size, "
            "this must not exceed it. Use 0 when the decision is to hold no "
            "position. Omit it only when no size can be stated at all; an omitted "
            "size is treated as unstated, not as zero."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description="Optional entry price in the instrument's quote currency.",
    )
    stop_loss: float | None = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )

    @field_validator("price_target", "position_size_pct", "entry_price",
                     "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    # No cross-field validator, for the same reason as TraderProposal: a rejected
    # proposal is discarded whole by invoke_structured_or_freetext's free-text
    # retry, so a validator that refused an out-of-limit size would destroy the
    # very number a compliance check needs to see. Faults are computed below.

    def levels(self) -> dict[str, object]:
        """The decision's numeric facts, for a compliance check and for a ledger.

        Mirrors :meth:`TraderProposal.levels`. Everything the model wrote passes
        through unchanged; ``None`` means unstated and never zero, because a
        decision that states no size is a different thing from one that recommends
        holding nothing.

        The size flags repeat ``TraderProposal``'s because the hazard is the same
        at both ends of the pipeline: ``0.05`` is a plausible way to write five
        percent and the two readings differ by 100x, so neither is assumed.
        """
        size = self.position_size_pct
        flags: list[str] = []
        if size is not None:
            if 0 < size <= 1:
                flags.append("size_ambiguous")
            elif size > 100:
                flags.append("size_out_of_range")
        rating = self.rating.value.lower()
        # A Buy with no position is a contradiction the reader would not catch:
        # the rating reads as an instruction and the size quietly cancels it.
        if size == 0 and rating in ("buy", "overweight"):
            flags.append("size_zero_on_a_buy")
        return {
            "rating": self.rating.value,
            "position_size_pct": size,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "price_target": self.price_target,
            "time_horizon": self.time_horizon or None,
            "flags": flags,
        }


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.position_size_pct is not None:
        parts.extend(["", f"**Position Size**: {decision.position_size_pct}% "
                          f"of portfolio"])
    if decision.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {decision.entry_price}"])
    if decision.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {decision.stop_loss}"])
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    flags = decision.levels()["flags"]
    if flags:
        parts.extend(["", "**Level Warnings**: " + ", ".join(
            _FLAG_TEXT.get(f, f) for f in flags)])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])


# ---------------------------------------------------------------------------
# Earnings & Estimate Revision Analyst
# ---------------------------------------------------------------------------


class EarningsNarrative(BaseModel):
    """Bounded qualitative synthesis layered over code-owned earnings evidence.

    This schema intentionally contains no consensus values, revision counts, or
    momentum label. Those fields are computed and rendered from the evidence
    tool result, so a language model cannot silently rewrite them.
    """

    guidance_and_commentary: str = Field(
        description=(
            "Sourced summary of management guidance and commentary present in "
            "the supplied evidence. Say 'Unavailable' when it is absent."
        ),
    )
    catalysts: list[str] = Field(
        default_factory=list,
        description="Up to five earnings-related catalysts supported by the evidence.",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Up to five earnings-related risks supported by the evidence.",
    )
    data_gaps: list[str] = Field(
        default_factory=list,
        description=(
            "Missing, stale, partial-coverage, or low-confidence fields. Preserve "
            "the provider's unavailable disclosures; never fill a gap by inference."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence in the qualitative synthesis only.",
    )

    @field_validator("catalysts", "risks", mode="after")
    @classmethod
    def _limit_bullets(cls, value: list[str]) -> list[str]:
        return value[:5]


def render_earnings_narrative(narrative: EarningsNarrative) -> str:
    """Render only qualitative fields; numeric evidence is rendered elsewhere."""

    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- Unavailable"

    return "\n".join([
        "## Guidance & Management Commentary",
        narrative.guidance_and_commentary or "Unavailable",
        "",
        "## Catalysts",
        bullets(narrative.catalysts),
        "",
        "## Risks",
        bullets(narrative.risks),
        "",
        "## Data Gaps",
        bullets(narrative.data_gaps),
        "",
        f"**Narrative Confidence:** {narrative.confidence.capitalize()}",
    ])

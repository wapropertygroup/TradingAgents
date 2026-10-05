"""Quality Analyst.

The thesis this agent encodes is Buffett/Munger's own framing: "is this a
good business" is a different question from "is this a good price", and
conflating them in one generic fundamentals report loses the sharper of the
two reads. This agent answers only the first question.

Built the same way ``earnings_analyst.py`` is, for the same reasons.

**The tool round is deterministic.** There is exactly one correct call
(ticker, curr_date); the model does not choose it.

**The numbers are rendered by code, and the model may only append prose.**
The published report is
``render_quality_report(evidence) + render_quality_narrative(narrative)``.
The model never emits a ratio or a tier -- it receives the finished numeric
report as context and is asked for a moat assessment, red flags and a
confidence rating. See :mod:`tradingagents.dataflows.quality_models` for the
scoring itself.

The LLM is skipped entirely when there is nothing to narrate -- a fund
wrapper (ETF, index) or a symbol with no fundamentals coverage yields a
report with no numbers in it.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.fundamentals_evidence_tools import get_quality_evidence
from tradingagents.agents.schemas import QualityNarrative, render_quality_narrative
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.quality_models import (
    QualityEvidence,
    render_quality_report,
)
from tradingagents.report_language import for_language

logger = logging.getLogger(__name__)

EVIDENCE_TOOL = "get_quality_evidence"

# The tools this analyst's first pass calls; its tool node is built from the same
# tuple. Called by code, not offered to the model, so none is bound to the LLM.
TOOLS = (get_quality_evidence,)

#: Statuses with no numeric surface to discuss. The LLM is not invoked for these.
_TERMINAL_STATUSES = {"unsupported", "no_coverage"}


def create_quality_analyst(llm):
    """Create the Quality Analyst node.

    Two passes through the same node, same shape as the Earnings Analyst:
    pass one emits the tool call and returns; the graph runs the ToolNode and
    re-enters. Pass two parses the result, asks for a bounded narrative, and
    renders the report.
    """
    structured_llm = bind_structured(llm, QualityNarrative, "Quality Analyst")

    def quality_analyst_node(state):
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        messages = state.get("messages") or []

        raw = _tool_result(messages)
        if raw is None:
            return {"messages": [_request_evidence(ticker, trade_date)]}

        evidence, parse_error = _parse_evidence(raw, ticker, trade_date)
        numeric_report = render_quality_report(evidence)
        if parse_error:
            numeric_report = f"{numeric_report}\n\n{parse_error}"

        if evidence.status in _TERMINAL_STATUSES:
            report = numeric_report
            return {"messages": [AIMessage(content=report)], "quality_report": report}

        narrative_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            _synthesis_prompt(
                ticker=ticker,
                trade_date=trade_date,
                instrument_context=get_instrument_context_from_state(state),
                numeric_report=numeric_report,
                evidence=evidence,
            ),
            render_quality_narrative,
            "Quality Analyst",
        )

        report = f"{numeric_report}\n\n{narrative_text}"
        return {"messages": [AIMessage(content=report)], "quality_report": report}

    return quality_analyst_node


# ---------------------------------------------------------------------------
# Pass one: the deterministic tool round
# ---------------------------------------------------------------------------


def _request_evidence(ticker: str, trade_date: str) -> AIMessage:
    """Emit the tool call directly, bypassing the model.

    A deterministic tool-call id, not a random one, so a checkpointed run
    resumes with the same message identity instead of a second, unmatched one.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": EVIDENCE_TOOL,
                "args": {"ticker": ticker, "curr_date": trade_date},
                "id": "quality_evidence_call",
            },
        ],
    )


def _tool_result(messages) -> str | None:
    """Latest content from this analyst's tool, from ToolMessages in this turn.

    Last-wins on a re-entered node, so a stale result from an earlier pass is
    never mistaken for the current one.
    """
    found: str | None = None
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "name", None) == EVIDENCE_TOOL:
            found = message.content if isinstance(message.content, str) else str(message.content)
    return found


# ---------------------------------------------------------------------------
# Pass two: parsing
# ---------------------------------------------------------------------------


def _parse_evidence(
    raw: str | None, ticker: str, trade_date: str
) -> tuple[QualityEvidence, str | None]:
    """Rehydrate the evidence, or synthesize an honest failure record."""
    L = for_language()
    if raw is None:
        return (
            QualityEvidence.no_coverage(
                ticker, trade_date,
                L("The quality evidence tool returned nothing for this run, so no "
                  "fundamentals figures are available.",
                  "本次运行中质量证据工具没有返回任何内容，因此没有可用的基本面数据。"),
            ),
            None,
        )

    text = raw.strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return (
            QualityEvidence.no_coverage(
                ticker, trade_date,
                L("The fundamentals data provider did not return structured evidence "
                  "for this symbol. Its verbatim response is reproduced below; treat "
                  "it as the reason, and do not supply figures from any other source.",
                  "基本面数据源没有为该代码返回结构化证据。其原始响应附在下方；请以此为原因，"
                  "不要用任何其他来源的数字替代。"),
            ),
            L("## Provider Response", "## 数据源原始响应") + f"\n\n```\n{text[:2000]}\n```",
        )

    try:
        return QualityEvidence.from_dict(payload), None
    except (ValueError, TypeError, AttributeError) as exc:
        logger.warning("quality evidence payload did not validate: %s", exc)
        return (
            QualityEvidence.no_coverage(
                ticker, trade_date,
                L(f"The quality evidence payload did not match the expected schema "
                  f"({exc}). No figures are reported.",
                  f"质量证据数据与预期结构不符（{exc}）。不报告任何数字。"),
            ),
            None,
        )


# ---------------------------------------------------------------------------
# Pass two: the bounded synthesis prompt
# ---------------------------------------------------------------------------


def _synthesis_prompt(
    *,
    ticker: str,
    trade_date: str,
    instrument_context: str,
    numeric_report: str,
    evidence: QualityEvidence,
) -> list:
    """Build a self-contained message list for the narrative pass.

    The graph's raw message history is deliberately not forwarded -- it holds
    an assistant turn with a tool call plus its JSON result, which the
    finished numeric report below already restates in readable form.
    """
    L = for_language()
    none = L("- none recorded", "- 无")
    gaps = "\n".join(f"- {L.message(gap)}" for gap in evidence.data_gaps) or none
    warnings = "\n".join(f"- {L.message(w)}" for w in evidence.warnings) or none

    system = f"""You are a business-quality analyst. A complete, already-\
finished numeric report for {ticker} as of {trade_date} is supplied below. \
It was computed from provider data by code.

{instrument_context}

Your job is ONLY to add the qualitative sections. You are filling three fields:
moat_assessment, red_flags, and a confidence rating.

## Hard constraints

1. **Do not restate, recompute, round, or correct any number.** Every ratio, \
percentage and the quality tier are already published in the report below and \
are appended verbatim ahead of your text. If you disagree with a value, say so \
in your reasoning; do not print a competing number.
2. **The quality tier is `{L.term(evidence.tier.tier)}` and is final.** It was computed \
from a weighted mean of the available signals. You may explain it or note that \
it rests on thin coverage. You may not upgrade, downgrade, or re-label it.
3. **Never fill a gap by inference.** Fields marked unavailable are absent from \
the source, not zero and not neutral.
4. **moat_assessment may draw on general knowledge of the business and its \
industry** -- unlike a consensus estimate, "does this have a durable moat" is \
not a number any vendor publishes. Say explicitly when you are reasoning from \
general knowledge rather than the figures above.
5. red_flags must be supported by the evidence above or by disclosed, reasoned \
business judgement (concentration, cyclicality, competitive threat, accounting \
quality). At most five.
6. data_gaps must preserve every gap and warning listed below, in your own \
words, plus anything else you find missing or internally inconsistent.
7. Your confidence rating describes YOUR qualitative synthesis only. It is not \
the tier's own signal coverage, which is already published.

## The finished numeric report (context — do not repeat it)

{numeric_report}

## Recorded data gaps

{gaps}

## Recorded warnings

{warnings}

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

    return [
        SystemMessage(content=system),
        HumanMessage(
            content=(
                f"Write the qualitative sections of the business-quality analysis "
                f"for {ticker} as of {trade_date}. Numbers are already handled; give "
                f"a moat assessment, red flags, data gaps, and your confidence in "
                f"the qualitative read."
            )
        ),
    ]

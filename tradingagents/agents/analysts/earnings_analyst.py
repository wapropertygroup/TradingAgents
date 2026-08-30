"""Earnings & Estimate Revision Analyst.

The thesis this agent encodes is that the tradable signal is usually not the
level of consensus EPS but its *direction*: whether the sell-side has been
marking a company up or down, how broadly, and how fast. So the report leads
with the fiscal-year consensus today against 7/30/90 days ago, the analyst
up/down counts behind that move, and a banded momentum verdict.

It is built differently from the other analysts, in two ways that matter.

**The tool round is deterministic.** The first pass issues the evidence and
commentary tool calls itself, with no LLM invocation at all. Every other analyst
binds tools and lets the model choose; that works when the model only has to
pick among plausible calls, but here there is exactly one correct call with
exactly one correct ``(ticker, curr_date)`` pair, and a model that forgets the
date or reaches for ``get_fundamentals`` instead produces a report that looks
complete and describes the wrong window. Doing it in code also removes one
round trip.

**The numbers are rendered by code, and the model may only append prose.** The
published report is
``render_evidence_report(evidence) + render_earnings_narrative(narrative)``.
The model never emits a figure, a band, or a date — it receives the finished
numeric report as context and is asked for guidance, catalysts, risks and gaps.
So a transcription slip or a confident round number cannot reach the reader,
and the momentum band in the report is always the one
:func:`~tradingagents.dataflows.earnings_models.compute_momentum` computed.

The corollary is that the LLM is skipped entirely when there is nothing to
narrate. An ETF, a symbol with no analyst coverage, or a historical date with no
stored vintage yields a report with no numbers in it; asking a model to write
five paragraphs of earnings analysis about that is an invitation to fill the
space from training data, and it would arrive indistinguishable from evidence.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tradingagents.agents.schemas import EarningsNarrative, render_earnings_narrative
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.earnings_models import (
    EarningsEvidence,
    render_evidence_report,
)

logger = logging.getLogger(__name__)

EVIDENCE_TOOL = "get_earnings_evidence"
COMMENTARY_TOOL = "get_earnings_commentary"

#: Statuses with no numeric surface to discuss. The LLM is not invoked for these.
_TERMINAL_STATUSES = {"unsupported", "pit_unavailable", "no_coverage"}

#: Transcript characters forwarded into the synthesis prompt. A full earnings
#: call runs to tens of thousands of tokens, which would dominate the context and
#: crowd out the numeric evidence the report is actually built from.
_COMMENTARY_BUDGET = 6000


def create_earnings_analyst(llm):
    """Create the Earnings Analyst node.

    Two passes through the same node. Pass one emits tool calls and returns; the
    graph runs the ToolNode and re-enters. Pass two parses the results, asks for a
    bounded narrative, and renders the report.
    """
    structured_llm = bind_structured(llm, EarningsNarrative, "Earnings Analyst")

    def earnings_analyst_node(state):
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        messages = state.get("messages") or []

        results = _tool_results(messages)
        if not results:
            return {"messages": [_request_evidence(ticker, trade_date)]}

        evidence, parse_error = _parse_evidence(results.get(EVIDENCE_TOOL), ticker, trade_date)
        commentary = _clean_commentary(results.get(COMMENTARY_TOOL))

        numeric_report = render_evidence_report(evidence)
        if parse_error:
            numeric_report = f"{numeric_report}\n\n{parse_error}"

        if evidence.status in _TERMINAL_STATUSES:
            # Nothing to narrate. Returning the deterministic report alone is the
            # whole answer, and it costs no tokens.
            report = numeric_report
            return {"messages": [AIMessage(content=report)], "earnings_report": report}

        narrative_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            _synthesis_prompt(
                ticker=ticker,
                trade_date=trade_date,
                instrument_context=get_instrument_context_from_state(state),
                numeric_report=numeric_report,
                commentary=commentary,
                evidence=evidence,
            ),
            render_earnings_narrative,
            "Earnings Analyst",
        )

        report = f"{numeric_report}\n\n{narrative_text}"
        return {"messages": [AIMessage(content=report)], "earnings_report": report}

    return earnings_analyst_node


# ---------------------------------------------------------------------------
# Pass one: the deterministic tool round
# ---------------------------------------------------------------------------


def _request_evidence(ticker: str, trade_date: str) -> AIMessage:
    """Emit both tool calls directly, bypassing the model.

    Tool-call ids are deterministic rather than random so a checkpointed run
    resumes with the same message identities instead of a second, unmatched pair.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": EVIDENCE_TOOL,
                "args": {"ticker": ticker, "curr_date": trade_date},
                "id": "earnings_evidence_call",
            },
            {
                "name": COMMENTARY_TOOL,
                "args": {"ticker": ticker, "curr_date": trade_date},
                "id": "earnings_commentary_call",
            },
        ],
    )


def _tool_results(messages) -> dict[str, str]:
    """Latest content per earnings tool, from ToolMessages in this turn.

    Keyed by tool name and last-wins, so a re-entered node reads the most recent
    result rather than a stale one. Only this analyst's two tools are collected —
    a preceding analyst's messages are cleared by ``Msg Clear`` before this node
    runs, but reading by name means an unexpected leftover cannot be mistaken for
    earnings evidence.
    """
    found: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        name = getattr(message, "name", None)
        if name in (EVIDENCE_TOOL, COMMENTARY_TOOL):
            found[name] = message.content if isinstance(message.content, str) else str(
                message.content
            )
    return found


# ---------------------------------------------------------------------------
# Pass two: parsing
# ---------------------------------------------------------------------------


def _parse_evidence(
    raw: str | None, ticker: str, trade_date: str
) -> tuple[EarningsEvidence, str | None]:
    """Rehydrate the evidence, or synthesize an honest failure record.

    The tool returns JSON on success and a prose sentinel on failure — the
    router's ``NO_DATA_AVAILABLE``/``DATA_UNAVAILABLE`` strings, or a ToolNode
    error message when a vendor raised. All three are non-JSON, so a parse
    failure is a normal outcome and is reported as one, carrying the sentinel
    verbatim so the reason survives into the report rather than being flattened
    to "unavailable".
    """
    if raw is None:
        return (
            EarningsEvidence.no_coverage(
                ticker, trade_date,
                "The earnings evidence tool returned nothing for this run, so no "
                "estimate, revision or surprise figures are available.",
            ),
            None,
        )

    text = raw.strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return (
            EarningsEvidence.no_coverage(
                ticker, trade_date,
                "The earnings data provider did not return structured evidence for "
                "this symbol. Its verbatim response is reproduced below; treat it as "
                "the reason, and do not supply figures from any other source.",
            ),
            f"## Provider Response\n\n```\n{text[:2000]}\n```",
        )

    try:
        return EarningsEvidence.from_dict(payload), None
    except (ValueError, TypeError, AttributeError) as exc:
        logger.warning("earnings evidence payload did not validate: %s", exc)
        return (
            EarningsEvidence.no_coverage(
                ticker, trade_date,
                f"The earnings evidence payload did not match the expected schema "
                f"({exc}). No figures are reported.",
            ),
            None,
        )


def _clean_commentary(raw: str | None) -> str | None:
    """The transcript, or ``None`` when the vendor declined.

    The unavailable markers are recognised and dropped rather than forwarded: a
    prompt containing "DATA_UNAVAILABLE: optional earnings_commentary could not be
    retrieved" invites the model to discuss the plumbing, and the absence is
    already stated in the report's data-gap section.
    """
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    for marker in (
        "EARNINGS_COMMENTARY_UNAVAILABLE",
        "DATA_UNAVAILABLE",
        "NO_DATA_AVAILABLE",
        "Error:",
    ):
        if text.startswith(marker):
            return None
    return text[:_COMMENTARY_BUDGET]


# ---------------------------------------------------------------------------
# Pass two: the bounded synthesis prompt
# ---------------------------------------------------------------------------


def _synthesis_prompt(
    *,
    ticker: str,
    trade_date: str,
    instrument_context: str,
    numeric_report: str,
    commentary: str | None,
    evidence: EarningsEvidence,
) -> list:
    """Build a self-contained message list for the narrative pass.

    The graph's raw message history is deliberately *not* forwarded. It holds an
    assistant turn with tool calls plus their results — thousands of tokens of
    JSON the model has no reason to re-read, since the finished numeric report
    below is the same information already formatted. Keeping the prompt
    self-contained also avoids provider-side tool-call/response pairing rules on
    a history this node assembled by hand.
    """
    commentary_block = (
        f"<earnings_call_transcript>\n{commentary}\n</earnings_call_transcript>"
        if commentary
        else (
            "<earnings_call_transcript>\nUNAVAILABLE — no earnings call transcript "
            "was retrieved for this run. Say so plainly in "
            "guidance_and_commentary; do not substitute news coverage, analyst "
            "notes, or recollection for what management said.\n"
            "</earnings_call_transcript>"
        )
    )

    gaps = "\n".join(f"- {gap}" for gap in evidence.data_gaps) or "- none recorded"
    warnings = "\n".join(f"- {w}" for w in evidence.warnings) or "- none recorded"

    system = f"""You are an earnings and estimate-revision analyst. A complete, \
already-finished numeric report for {ticker} as of {trade_date} is supplied below. \
It was computed from provider data by code.

{instrument_context}

Your job is ONLY to add the qualitative sections. You are filling four fields:
guidance_and_commentary, catalysts, risks, and data_gaps, plus a confidence rating.

## Hard constraints

1. **Do not restate, recompute, round, or correct any number.** Every figure, \
percentage, count, date and the momentum band are already published in the report \
below and are appended verbatim ahead of your text. If you disagree with a value, \
say so in data_gaps in words; do not print a competing number.
2. **The momentum band is `{evidence.momentum.band}` and is final.** It was computed \
from a weighted mean of the available revision signals. You may explain it or note \
that it rests on thin coverage. You may not upgrade, downgrade, or re-label it.
3. **Never fill a gap by inference.** Fields marked unavailable are absent from the \
source, not zero and not neutral. Whisper expectations, consensus margin revisions \
and 90-day revision breadth in particular are unavailable by design — do not \
estimate them, and do not present news sentiment as a whisper number.
4. **Guidance means sourced guidance.** Only describe management guidance or \
commentary that appears in the transcript block below. If it is unavailable, write \
"Unavailable" and stop. Do not infer guidance from an estimate revision — analysts \
revising a number is not the company saying anything.
5. Catalysts and risks must be earnings-related and supported by what is in this \
prompt: the next reporting date, the revision direction and its breadth, the \
surprise record, the drift record, or sourced commentary. At most five of each.
6. data_gaps must preserve every gap and warning listed below, in your own words, \
plus anything else you find missing or internally inconsistent.
7. Your confidence rating describes YOUR qualitative synthesis only. It is not the \
momentum confidence, which is already published.

## The finished numeric report (context — do not repeat it)

{numeric_report}

## Recorded data gaps

{gaps}

## Recorded warnings

{warnings}

## Management commentary

{commentary_block}

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

    return [
        SystemMessage(content=system),
        HumanMessage(
            content=(
                f"Write the qualitative sections of the earnings and estimate-revision "
                f"analysis for {ticker} as of {trade_date}. Numbers are already handled; "
                f"give guidance and commentary, catalysts, risks, data gaps, and your "
                f"confidence in the qualitative read."
            )
        ),
    ]

"""Valuation Analyst.

The other half of the fundamentals split: "is this a good price", independent
of whether it is a good business. Built the same way ``earnings_analyst.py``
and ``quality_analyst.py`` are -- see ``quality_analyst.py``'s module
docstring for the shared design rationale (deterministic tool round, code-
owned numbers, LLM narrates only, skipped entirely with nothing to narrate).
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.fundamentals_evidence_tools import get_valuation_evidence
from tradingagents.agents.schemas import ValuationNarrative, render_valuation_narrative
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.valuation_models import (
    ValuationEvidence,
    render_valuation_report,
)
from tradingagents.report_language import for_language

logger = logging.getLogger(__name__)

EVIDENCE_TOOL = "get_valuation_evidence"

# The tools this analyst's first pass calls; its tool node is built from the same
# tuple. Called by code, not offered to the model, so none is bound to the LLM.
TOOLS = (get_valuation_evidence,)

_TERMINAL_STATUSES = {"unsupported", "no_coverage"}


def create_valuation_analyst(llm):
    """Create the Valuation Analyst node. Two-pass, same shape as Quality's."""
    structured_llm = bind_structured(llm, ValuationNarrative, "Valuation Analyst")

    def valuation_analyst_node(state):
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        messages = state.get("messages") or []

        raw = _tool_result(messages)
        if raw is None:
            return {"messages": [_request_evidence(ticker, trade_date)]}

        evidence, parse_error = _parse_evidence(raw, ticker, trade_date)
        numeric_report = render_valuation_report(evidence)
        if parse_error:
            numeric_report = f"{numeric_report}\n\n{parse_error}"

        if evidence.status in _TERMINAL_STATUSES:
            report = numeric_report
            return {"messages": [AIMessage(content=report)], "valuation_report": report}

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
            render_valuation_narrative,
            "Valuation Analyst",
        )

        report = f"{numeric_report}\n\n{narrative_text}"
        return {"messages": [AIMessage(content=report)], "valuation_report": report}

    return valuation_analyst_node


def _request_evidence(ticker: str, trade_date: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": EVIDENCE_TOOL,
                "args": {"ticker": ticker, "curr_date": trade_date},
                "id": "valuation_evidence_call",
            },
        ],
    )


def _tool_result(messages) -> str | None:
    found: str | None = None
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "name", None) == EVIDENCE_TOOL:
            found = message.content if isinstance(message.content, str) else str(message.content)
    return found


def _parse_evidence(
    raw: str | None, ticker: str, trade_date: str
) -> tuple[ValuationEvidence, str | None]:
    L = for_language()
    if raw is None:
        return (
            ValuationEvidence.no_coverage(
                ticker, trade_date,
                L("The valuation evidence tool returned nothing for this run, so no "
                  "multiples are available.",
                  "本次运行中估值证据工具没有返回任何内容，因此没有可用的估值倍数。"),
            ),
            None,
        )

    text = raw.strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return (
            ValuationEvidence.no_coverage(
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
        return ValuationEvidence.from_dict(payload), None
    except (ValueError, TypeError, AttributeError) as exc:
        logger.warning("valuation evidence payload did not validate: %s", exc)
        return (
            ValuationEvidence.no_coverage(
                ticker, trade_date,
                L(f"The valuation evidence payload did not match the expected schema "
                  f"({exc}). No figures are reported.",
                  f"估值证据数据与预期结构不符（{exc}）。不报告任何数字。"),
            ),
            None,
        )


def _synthesis_prompt(
    *,
    ticker: str,
    trade_date: str,
    instrument_context: str,
    numeric_report: str,
    evidence: ValuationEvidence,
) -> list:
    L = for_language()
    none = L("- none recorded", "- 无")
    gaps = "\n".join(f"- {L.message(gap)}" for gap in evidence.data_gaps) or none
    warnings = "\n".join(f"- {L.message(w)}" for w in evidence.warnings) or none

    system = f"""You are a valuation analyst. A complete, already-finished \
numeric report for {ticker} as of {trade_date} is supplied below. It was \
computed from provider data by code.

{instrument_context}

Your job is ONLY to add the qualitative sections. You are filling three fields:
thesis, catalysts_for_rerating, and a confidence rating.

## Hard constraints

1. **Do not restate, recompute, round, or correct any number.** Every multiple \
and the valuation tier are already published in the report below and are \
appended verbatim ahead of your text.
2. **The valuation tier is `{L.term(evidence.tier.tier)}` and is final.** It was \
computed from a weighted mean of the available signals (P/E band, PEG, \
price-to-book, EV/EBIT, EV/Sales, forward-vs-trailing P/E, dividend yield). \
EV/Sales is the one multiple a loss-making company still has. You may explain it — \
including why a single headline multiple might look different from the tier — \
but you may not upgrade, downgrade, or re-label it.
3. **Never fill a gap by inference.** A missing trailing P/E is commonly a \
negative-earnings company; it is absent, not zero, and not evidence the stock \
is either cheap or expensive.
4. **thesis may draw on general knowledge of the business's growth trajectory** \
to explain why the tier looks the way it does, but must not re-derive the tier \
itself in different words.
5. catalysts_for_rerating must be supported by the evidence above or by \
disclosed, reasoned business judgement. At most five.
6. data_gaps must preserve every gap and warning listed below, in your own \
words, plus anything else you find missing or internally inconsistent.
7. Your confidence rating describes YOUR qualitative synthesis only.

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
                f"Write the qualitative sections of the valuation analysis for "
                f"{ticker} as of {trade_date}. Numbers are already handled; give a "
                f"thesis, catalysts for re-rating, data gaps, and your confidence "
                f"in the qualitative read."
            )
        ),
    ]

"""Quality Analyst: deterministic tool round, then narrative-only synthesis.

Same contract under test as ``test_earnings_analyst.py``: the language model
can add prose and cannot touch a number. Both passes run against a fake LLM.
"""

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tradingagents.agents.analysts.quality_analyst import (
    EVIDENCE_TOOL,
    create_quality_analyst,
)
from tradingagents.agents.schemas import QualityNarrative
from tradingagents.dataflows.quality_models import QualityEvidence, Value, finalize_evidence

NARRATIVE = QualityNarrative(
    moat_assessment="Durable brand and ecosystem lock-in support pricing power.",
    red_flags=["Customer concentration in one product line"],
    data_gaps=["Fewer than three years of margin history available"],
    confidence="medium",
)


class FakeLLM:
    """Captures prompts and returns a canned structured narrative."""

    def __init__(self, *, structured=True, narrative=NARRATIVE, freetext="FREE TEXT BODY"):
        self.prompts = []
        self.structured_prompts = []
        self._narrative = narrative
        self._freetext = freetext
        self._supports_structured = structured

    def with_structured_output(self, _schema):
        if not self._supports_structured:
            raise NotImplementedError("provider lacks structured output")
        outer = self

        class Bound:
            def invoke(self, prompt):
                outer.structured_prompts.append(prompt)
                if isinstance(outer._narrative, Exception):
                    raise outer._narrative
                return outer._narrative

        return Bound()

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self._freetext)

    def all_prompt_text(self):
        parts = []
        for prompt in list(self.prompts) + list(self.structured_prompts):
            if isinstance(prompt, list):
                parts.extend(getattr(m, "content", str(m)) for m in prompt)
            else:
                parts.append(str(prompt))
        return "\n".join(parts)


def _evidence(**overrides):
    base = {
        "symbol": "AAPL", "as_of": "2026-08-30", "company_name": "Apple Inc.",
        "currency": "USD",
        "return_on_equity": Value(1.4875, unit="pct_dec"),
        "operating_margin": Value(0.326, unit="pct_dec"),
        "debt_to_equity": Value(0.784, unit="ratio"),
        "current_ratio": Value(1.003, unit="ratio"),
        "free_cash_flow": Value(107721875456, unit="currency_large", currency="USD"),
        "total_revenue": Value(466822987776, unit="currency_large", currency="USD"),
        "sources": ["yfinance (Yahoo Finance fundamentals)"],
    }
    base.update(overrides)
    return finalize_evidence(QualityEvidence(**base))


def _state(*, messages=None):
    return {
        "company_of_interest": "AAPL",
        "trade_date": "2026-08-30",
        "asset_type": "stock",
        "instrument_context": "The instrument to analyze is `AAPL`.",
        "messages": messages if messages is not None else [HumanMessage(content="go")],
    }


def _tool_messages(evidence):
    return [
        HumanMessage(content="go"),
        ToolMessage(
            content=json.dumps(evidence.to_dict()),
            name=EVIDENCE_TOOL, tool_call_id="quality_evidence_call",
        ),
    ]


class FirstPassTests(unittest.TestCase):
    def test_the_tool_is_called_with_the_exact_ticker_and_date(self):
        llm = FakeLLM()
        out = create_quality_analyst(llm)(_state())
        calls = out["messages"][0].tool_calls
        self.assertEqual({c["name"] for c in calls}, {EVIDENCE_TOOL})
        self.assertEqual(calls[0]["args"], {"ticker": "AAPL", "curr_date": "2026-08-30"})

    def test_no_llm_call_is_made_on_the_first_pass(self):
        llm = FakeLLM()
        create_quality_analyst(llm)(_state())
        self.assertEqual(llm.prompts, [])
        self.assertEqual(llm.structured_prompts, [])

    def test_no_report_is_written_on_the_first_pass(self):
        out = create_quality_analyst(FakeLLM())(_state())
        self.assertNotIn("quality_report", out)

    def test_tool_call_id_is_deterministic(self):
        first = create_quality_analyst(FakeLLM())(_state())["messages"][0]
        second = create_quality_analyst(FakeLLM())(_state())["messages"][0]
        self.assertEqual(
            [c["id"] for c in first.tool_calls], [c["id"] for c in second.tool_calls]
        )


class SecondPassTests(unittest.TestCase):
    def setUp(self):
        self.evidence = _evidence()
        self.llm = FakeLLM()
        self.out = create_quality_analyst(self.llm)(
            _state(messages=_tool_messages(self.evidence))
        )
        self.report = self.out["quality_report"]

    def test_the_report_is_written_to_state_and_messages(self):
        self.assertTrue(self.report)
        self.assertEqual(self.out["messages"][0].content, self.report)

    def test_the_computed_tier_appears_and_the_model_did_not_choose_it(self):
        self.assertIn(f"**{self.evidence.tier.tier}**", self.report)

    def test_numbers_come_from_the_evidence_not_the_narrative(self):
        self.assertIn("+148.75%", self.report)
        self.assertIn("+0.784", self.report)

    def test_the_narrative_sections_are_appended(self):
        for heading in ("## Moat & Competitive Positioning", "## Red Flags", "## Data Gaps"):
            self.assertIn(heading, self.report)
        self.assertIn("ecosystem lock-in", self.report)
        self.assertIn("**Narrative Confidence:** Medium", self.report)

    def test_numeric_sections_precede_the_narrative(self):
        self.assertLess(
            self.report.index("## Quality Tier"),
            self.report.index("## Moat & Competitive Positioning"),
        )

    def test_the_prompt_forbids_restating_or_relabelling_the_tier(self):
        text = self.llm.all_prompt_text()
        self.assertIn("Do not restate, recompute, round, or correct any number", text)
        self.assertIn("is final", text)
        self.assertIn("Never fill a gap by inference", text)

    def test_the_prompt_states_the_computed_tier_so_it_cannot_be_re_derived(self):
        self.assertIn(f"quality tier is `{self.evidence.tier.tier}`", self.llm.all_prompt_text())

    def test_the_prompt_permits_general_business_knowledge_for_the_moat(self):
        self.assertIn("may draw on general knowledge", self.llm.all_prompt_text())


class TerminalStatusTests(unittest.TestCase):
    def _run(self, evidence):
        llm = FakeLLM()
        out = create_quality_analyst(llm)(_state(messages=_tool_messages(evidence)))
        return llm, out["quality_report"]

    def test_a_fund_wrapper_skips_the_model_entirely(self):
        llm, report = self._run(
            QualityEvidence.unsupported("SPY", "2026-08-30", "SPY is an etf.")
        )
        self.assertEqual(llm.prompts, [])
        self.assertEqual(llm.structured_prompts, [])
        self.assertIn("not applicable", report)
        self.assertNotIn("## Red Flags", report)

    def test_no_coverage_skips_the_model(self):
        llm, report = self._run(
            QualityEvidence.no_coverage("XYZ", "2026-08-30", "No fundamentals coverage.")
        )
        self.assertEqual(llm.structured_prompts, [])
        self.assertIn("No fundamentals coverage", report)


class StructuredFallbackTests(unittest.TestCase):
    def test_a_provider_without_structured_output_still_produces_a_report(self):
        llm = FakeLLM(structured=False, freetext="Some qualitative prose.")
        report = create_quality_analyst(llm)(
            _state(messages=_tool_messages(_evidence()))
        )["quality_report"]
        self.assertIn("Some qualitative prose.", report)

    def test_the_computed_numbers_survive_a_structured_output_failure(self):
        llm = FakeLLM(narrative=ValueError("malformed json"),
                      freetext="I think this is High Quality with ROE of 200%.")
        evidence = _evidence()
        report = create_quality_analyst(llm)(
            _state(messages=_tool_messages(evidence))
        )["quality_report"]
        self.assertIn(f"**{evidence.tier.tier}**", report)
        self.assertLess(report.index(f"**{evidence.tier.tier}**"),
                        report.index("High Quality with ROE"))


if __name__ == "__main__":
    unittest.main()

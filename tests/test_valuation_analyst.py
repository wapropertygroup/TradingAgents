"""Valuation Analyst: deterministic tool round, then narrative-only synthesis.

Same contract under test as ``test_quality_analyst.py`` / ``test_earnings_analyst.py``.
"""

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tradingagents.agents.analysts.valuation_analyst import (
    EVIDENCE_TOOL,
    create_valuation_analyst,
)
from tradingagents.agents.schemas import ValuationNarrative
from tradingagents.dataflows.valuation_models import ValuationEvidence, Value, finalize_evidence

NARRATIVE = ValuationNarrative(
    thesis="Fair despite an expensive headline multiple because PEG and the "
          "forward/trailing spread price in continued growth.",
    catalysts_for_rerating=["Next earnings beat could compress the forward multiple"],
    data_gaps=[],
    confidence="medium",
)


class FakeLLM:
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
        "symbol": "NVDA", "as_of": "2026-08-30", "company_name": "NVIDIA Corp.",
        "currency": "USD",
        "trailing_pe": Value(28.81, unit="ratio"),
        "forward_pe": Value(14.21, unit="ratio"),
        "peg_ratio": Value(0.63, unit="ratio"),
        "price_to_book": Value(26.96, unit="ratio"),
        "dividend_yield": Value(0.0044, unit="pct_dec"),
        "sources": ["yfinance (Yahoo Finance fundamentals)"],
    }
    base.update(overrides)
    return finalize_evidence(ValuationEvidence(**base))


def _state(*, messages=None):
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-08-30",
        "asset_type": "stock",
        "instrument_context": "The instrument to analyze is `NVDA`.",
        "messages": messages if messages is not None else [HumanMessage(content="go")],
    }


def _tool_messages(evidence):
    return [
        HumanMessage(content="go"),
        ToolMessage(
            content=json.dumps(evidence.to_dict()),
            name=EVIDENCE_TOOL, tool_call_id="valuation_evidence_call",
        ),
    ]


class FirstPassTests(unittest.TestCase):
    def test_the_tool_is_called_with_the_exact_ticker_and_date(self):
        llm = FakeLLM()
        out = create_valuation_analyst(llm)(_state())
        calls = out["messages"][0].tool_calls
        self.assertEqual({c["name"] for c in calls}, {EVIDENCE_TOOL})
        self.assertEqual(calls[0]["args"], {"ticker": "NVDA", "curr_date": "2026-08-30"})

    def test_no_llm_call_is_made_on_the_first_pass(self):
        llm = FakeLLM()
        create_valuation_analyst(llm)(_state())
        self.assertEqual(llm.prompts, [])
        self.assertEqual(llm.structured_prompts, [])

    def test_no_report_is_written_on_the_first_pass(self):
        out = create_valuation_analyst(FakeLLM())(_state())
        self.assertNotIn("valuation_report", out)


class SecondPassTests(unittest.TestCase):
    def setUp(self):
        self.evidence = _evidence()
        self.llm = FakeLLM()
        self.out = create_valuation_analyst(self.llm)(
            _state(messages=_tool_messages(self.evidence))
        )
        self.report = self.out["valuation_report"]

    def test_the_report_is_written_to_state_and_messages(self):
        self.assertTrue(self.report)
        self.assertEqual(self.out["messages"][0].content, self.report)

    def test_the_computed_tier_appears_and_the_model_did_not_choose_it(self):
        self.assertIn(f"**{self.evidence.tier.tier}**", self.report)

    def test_numbers_come_from_the_evidence_not_the_narrative(self):
        self.assertIn("28.810", self.report)
        self.assertIn("0.630", self.report)

    def test_the_narrative_sections_are_appended(self):
        for heading in ("## Valuation Thesis", "## Catalysts for Re-rating", "## Data Gaps"):
            self.assertIn(heading, self.report)
        self.assertIn("forward/trailing spread", self.report)

    def test_numeric_sections_precede_the_narrative(self):
        self.assertLess(
            self.report.index("## Valuation Tier"),
            self.report.index("## Valuation Thesis"),
        )

    def test_the_prompt_forbids_restating_or_relabelling_the_tier(self):
        text = self.llm.all_prompt_text()
        self.assertIn("Do not restate, recompute, round, or correct any number", text)
        self.assertIn("is final", text)

    def test_the_prompt_states_the_computed_tier_so_it_cannot_be_re_derived(self):
        self.assertIn(f"valuation tier is `{self.evidence.tier.tier}`", self.llm.all_prompt_text())

    def test_the_prompt_warns_against_treating_missing_pe_as_cheap_or_expensive(self):
        self.assertIn("not evidence the stock is either cheap or expensive",
                      self.llm.all_prompt_text())


class TerminalStatusTests(unittest.TestCase):
    def _run(self, evidence):
        llm = FakeLLM()
        out = create_valuation_analyst(llm)(_state(messages=_tool_messages(evidence)))
        return llm, out["valuation_report"]

    def test_a_fund_wrapper_skips_the_model_entirely(self):
        llm, report = self._run(
            ValuationEvidence.unsupported("SPY", "2026-08-30", "SPY is an etf.")
        )
        self.assertEqual(llm.prompts, [])
        self.assertEqual(llm.structured_prompts, [])
        self.assertIn("not applicable", report)

    def test_no_coverage_skips_the_model(self):
        llm, report = self._run(
            ValuationEvidence.no_coverage("XYZ", "2026-08-30", "No valuation coverage.")
        )
        self.assertEqual(llm.structured_prompts, [])
        self.assertIn("No valuation coverage", report)


class StructuredFallbackTests(unittest.TestCase):
    def test_the_computed_numbers_survive_a_structured_output_failure(self):
        llm = FakeLLM(narrative=ValueError("malformed json"),
                      freetext="I think this is Deep Value at a 5x P/E.")
        evidence = _evidence()
        report = create_valuation_analyst(llm)(
            _state(messages=_tool_messages(evidence))
        )["valuation_report"]
        self.assertIn(f"**{evidence.tier.tier}**", report)
        self.assertLess(report.index(f"**{evidence.tier.tier}**"),
                        report.index("Deep Value at a 5x"))


if __name__ == "__main__":
    unittest.main()

"""Earnings Analyst: deterministic tool round, then narrative-only synthesis.

The contract under test is that the language model can add prose and cannot
touch a number. Both passes run against a fake LLM, so nothing here needs a
provider.
"""

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tradingagents.agents.analysts.earnings_analyst import (
    COMMENTARY_TOOL,
    EVIDENCE_TOOL,
    create_earnings_analyst,
)
from tradingagents.agents.schemas import EarningsNarrative
from tradingagents.dataflows.earnings_models import (
    EarningsCalendar,
    EarningsEvidence,
    EstimateTrend,
    FiscalPeriod,
    PeriodEvidence,
    RevisionBreadth,
    SurpriseEvent,
    Value,
    finalize_evidence,
)

NARRATIVE = EarningsNarrative(
    guidance_and_commentary="Unavailable — no transcript was retrieved.",
    catalysts=["FY2026 report due 2026-10-29"],
    risks=["Seven-day breadth is inconsistent with the thirty-day counts"],
    data_gaps=["No revenue revision history is published"],
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
    period = PeriodEvidence(
        period=FiscalPeriod(key="0y", end_date="2026-09-27"),
        eps=EstimateTrend(
            current=Value(8.81249, currency="USD"),
            days_ago_7=Value(8.80708, currency="USD"),
            days_ago_30=Value(8.76760, currency="USD"),
            days_ago_60=Value(8.75958, currency="USD"),
            days_ago_90=Value(8.75324, currency="USD"),
        ),
        revenue=EstimateTrend(
            current=Value(477683718840, unit="currency_large", currency="USD"),
        ),
        breadth=RevisionBreadth(
            up_7d=Value(20, unit="count"), down_7d=Value(9, unit="count"),
            up_30d=Value(21, unit="count"), down_30d=Value(8, unit="count"),
        ),
        analyst_count=Value(37, unit="count"),
    )
    base = dict(
        symbol="AAPL", as_of="2026-08-30", company_name="Apple Inc.", currency="USD",
        periods={"0y": period},
        calendar=EarningsCalendar(
            next_date="2026-10-29", eps_estimate_avg=Value(1.98013, currency="USD")
        ),
        surprises=[SurpriseEvent(
            fiscal_period_end="2026-06-30", eps_actual=Value(2.02),
            eps_estimate=Value(1.89243), eps_difference=Value(0.13),
            surprise_pct=Value(0.0674, unit="pct_dec"),
        )],
        sources=["yfinance"],
    )
    base.update(overrides)
    return finalize_evidence(EarningsEvidence(**base))


def _state(*, messages=None):
    return {
        "company_of_interest": "AAPL",
        "trade_date": "2026-08-30",
        "asset_type": "stock",
        "instrument_context": "The instrument to analyze is `AAPL`.",
        "messages": messages if messages is not None else [HumanMessage(content="go")],
    }


def _tool_messages(evidence, commentary="DATA_UNAVAILABLE: optional earnings_commentary"):
    return [
        HumanMessage(content="go"),
        ToolMessage(
            content=json.dumps(evidence.to_dict()),
            name=EVIDENCE_TOOL, tool_call_id="earnings_evidence_call",
        ),
        ToolMessage(
            content=commentary, name=COMMENTARY_TOOL,
            tool_call_id="earnings_commentary_call",
        ),
    ]


class FirstPassTests(unittest.TestCase):
    """The tool round is issued by code, not chosen by the model."""

    def test_both_tools_are_called_with_the_exact_ticker_and_date(self):
        llm = FakeLLM()
        out = create_earnings_analyst(llm)(_state())
        calls = out["messages"][0].tool_calls
        self.assertEqual(
            {c["name"] for c in calls}, {EVIDENCE_TOOL, COMMENTARY_TOOL}
        )
        for call in calls:
            self.assertEqual(call["args"], {"ticker": "AAPL", "curr_date": "2026-08-30"})

    def test_no_llm_call_is_made_on_the_first_pass(self):
        llm = FakeLLM()
        create_earnings_analyst(llm)(_state())
        self.assertEqual(llm.prompts, [])
        self.assertEqual(llm.structured_prompts, [])

    def test_no_report_is_written_on_the_first_pass(self):
        out = create_earnings_analyst(FakeLLM())(_state())
        self.assertNotIn("earnings_report", out)

    def test_tool_call_ids_are_deterministic_so_a_resume_matches(self):
        first = create_earnings_analyst(FakeLLM())(_state())["messages"][0]
        second = create_earnings_analyst(FakeLLM())(_state())["messages"][0]
        self.assertEqual(
            [c["id"] for c in first.tool_calls], [c["id"] for c in second.tool_calls]
        )

    def test_a_foreign_tool_message_does_not_count_as_earnings_evidence(self):
        """Results are collected by tool name, not by position."""
        state = _state(messages=[
            HumanMessage(content="go"),
            ToolMessage(content="some news", name="get_news", tool_call_id="x"),
        ])
        out = create_earnings_analyst(FakeLLM())(state)
        self.assertTrue(out["messages"][0].tool_calls)


class SecondPassTests(unittest.TestCase):
    def setUp(self):
        self.evidence = _evidence()
        self.llm = FakeLLM()
        self.out = create_earnings_analyst(self.llm)(
            _state(messages=_tool_messages(self.evidence))
        )
        self.report = self.out["earnings_report"]

    def test_the_report_is_written_to_state_and_messages(self):
        self.assertTrue(self.report)
        self.assertEqual(self.out["messages"][0].content, self.report)

    def test_the_computed_band_appears_and_the_model_did_not_choose_it(self):
        self.assertEqual(self.evidence.momentum.band, "Neutral")
        self.assertIn("**Neutral**", self.report)

    def test_numbers_come_from_the_evidence_not_the_narrative(self):
        self.assertIn("8.81 USD", self.report)
        self.assertIn("+21 raised / -8 lowered", self.report)
        self.assertIn("Analysts covering this period: 37", self.report)

    def test_the_narrative_sections_are_appended(self):
        for heading in (
            "## Guidance & Management Commentary", "## Catalysts", "## Risks", "## Data Gaps"
        ):
            self.assertIn(heading, self.report)
        self.assertIn("FY2026 report due 2026-10-29", self.report)
        self.assertIn("**Narrative Confidence:** Medium", self.report)

    def test_numeric_sections_precede_the_narrative(self):
        self.assertLess(
            self.report.index("## Earnings Momentum"),
            self.report.index("## Catalysts"),
        )

    def test_the_prompt_forbids_restating_or_relabelling_figures(self):
        text = self.llm.all_prompt_text()
        self.assertIn("Do not restate, recompute, round, or correct any number", text)
        self.assertIn("is final", text)
        self.assertIn("Never fill a gap by inference", text)

    def test_the_prompt_states_the_computed_band_so_it_cannot_be_re_derived(self):
        self.assertIn("momentum band is `Neutral`", self.llm.all_prompt_text())

    def test_the_prompt_carries_the_recorded_gaps_and_warnings(self):
        text = self.llm.all_prompt_text()
        self.assertIn("Recorded data gaps", text)
        self.assertIn("Whisper expectations", text)

    def test_the_raw_tool_history_is_not_forwarded_to_the_model(self):
        """Thousands of tokens of JSON the model has no reason to re-read."""
        text = self.llm.all_prompt_text()
        self.assertNotIn('"schema_version"', text)
        self.assertNotIn('"unavailable_reason"', text)

    def test_the_commentary_sentinel_is_not_forwarded_as_content(self):
        """A plumbing message in the prompt invites prose about the plumbing."""
        text = self.llm.all_prompt_text()
        self.assertNotIn("DATA_UNAVAILABLE", text)
        self.assertIn("UNAVAILABLE — no earnings call transcript", text)


class CommentaryTests(unittest.TestCase):
    def test_a_real_transcript_reaches_the_prompt(self):
        llm = FakeLLM()
        transcript = (
            "# Earnings call transcript — AAPL 2026Q2\n\n"
            "**Tim Cook — CEO**\n\nWe expect September quarter revenue to grow."
        )
        create_earnings_analyst(llm)(
            _state(messages=_tool_messages(_evidence(), commentary=transcript))
        )
        text = llm.all_prompt_text()
        self.assertIn("September quarter revenue to grow", text)
        self.assertIn("<earnings_call_transcript>", text)

    def test_every_unavailable_marker_is_recognised(self):
        for marker in (
            "EARNINGS_COMMENTARY_UNAVAILABLE: nope",
            "DATA_UNAVAILABLE: optional earnings_commentary could not be retrieved",
            "NO_DATA_AVAILABLE: no usable market data",
            "Error: tool failed",
            "", "   ",
        ):
            with self.subTest(marker=marker[:30]):
                llm = FakeLLM()
                create_earnings_analyst(llm)(
                    _state(messages=_tool_messages(_evidence(), commentary=marker))
                )
                self.assertIn(
                    "UNAVAILABLE — no earnings call transcript", llm.all_prompt_text()
                )

    def test_a_long_transcript_is_truncated_so_it_cannot_crowd_out_the_evidence(self):
        llm = FakeLLM()
        create_earnings_analyst(llm)(
            _state(messages=_tool_messages(_evidence(), commentary="word " * 20000))
        )
        text = llm.all_prompt_text()
        self.assertLess(text.count("word"), 3000)
        self.assertIn("## Earnings Momentum", text)


class TerminalStatusTests(unittest.TestCase):
    """No numbers to discuss means no LLM call at all."""

    def _run(self, evidence):
        llm = FakeLLM()
        out = create_earnings_analyst(llm)(_state(messages=_tool_messages(evidence)))
        return llm, out["earnings_report"]

    def test_an_etf_skips_the_model_entirely(self):
        llm, report = self._run(
            EarningsEvidence.unsupported("SPY", "2026-08-30", "SPY is an etf.")
        )
        self.assertEqual(llm.prompts, [])
        self.assertEqual(llm.structured_prompts, [])
        self.assertIn("not applicable", report)
        self.assertNotIn("## Catalysts", report)

    def test_a_missing_vintage_skips_the_model(self):
        llm, report = self._run(
            EarningsEvidence.pit_unavailable("AAPL", "2020-01-15", "No snapshot observed.")
        )
        self.assertEqual(llm.structured_prompts, [])
        self.assertIn("Point-in-time earnings evidence unavailable", report)

    def test_no_coverage_skips_the_model(self):
        llm, report = self._run(
            EarningsEvidence.no_coverage("XYZ", "2026-08-30", "No sell-side coverage.")
        )
        self.assertEqual(llm.structured_prompts, [])
        self.assertIn("No analyst estimate coverage", report)


class ToolFailureTests(unittest.TestCase):
    def _report(self, evidence_content):
        llm = FakeLLM()
        messages = [
            HumanMessage(content="go"),
            ToolMessage(content=evidence_content, name=EVIDENCE_TOOL, tool_call_id="a"),
        ]
        out = create_earnings_analyst(llm)(_state(messages=messages))
        return llm, out["earnings_report"]

    def test_a_router_sentinel_is_reproduced_verbatim_as_the_reason(self):
        sentinel = (
            "NO_DATA_AVAILABLE: No usable market data for '600519' from any "
            "configured vendor (bare numeric symbol is not a Yahoo ticker)."
        )
        llm, report = self._report(sentinel)
        self.assertIn("did not return structured evidence", report)
        self.assertIn("bare numeric symbol", report)
        self.assertEqual(llm.structured_prompts, [], "no narrative for a terminal status")

    def test_a_tool_node_error_string_is_handled_not_parsed(self):
        _llm, report = self._report("Error: KeyError('epsActual')")
        self.assertIn("did not return structured evidence", report)
        self.assertIn("KeyError", report)

    def test_a_valid_json_payload_of_the_wrong_shape_is_reported(self):
        _llm, report = self._report(json.dumps(["not", "an", "object"]))
        self.assertIn("No analyst estimate coverage", report)

    def test_a_missing_evidence_tool_result_is_reported(self):
        llm = FakeLLM()
        messages = [
            HumanMessage(content="go"),
            ToolMessage(content="whatever", name=COMMENTARY_TOOL, tool_call_id="b"),
        ]
        report = create_earnings_analyst(llm)(_state(messages=messages))["earnings_report"]
        self.assertIn("returned nothing for this run", report)


class StructuredFallbackTests(unittest.TestCase):
    def test_a_provider_without_structured_output_still_produces_a_report(self):
        llm = FakeLLM(structured=False, freetext="Some qualitative prose.")
        report = create_earnings_analyst(llm)(
            _state(messages=_tool_messages(_evidence()))
        )["earnings_report"]
        self.assertIn("Some qualitative prose.", report)

    def test_the_computed_numbers_survive_a_structured_output_failure(self):
        """A free-text fallback must not be able to change a figure or the band."""
        llm = FakeLLM(narrative=ValueError("malformed json"),
                      freetext="I think momentum is Strong Positive and EPS is 99.99.")
        report = create_earnings_analyst(llm)(
            _state(messages=_tool_messages(_evidence()))
        )["earnings_report"]
        # The code-owned header still says Neutral and 8.81.
        self.assertIn("**Neutral**", report)
        self.assertIn("8.81 USD", report)
        self.assertLess(report.index("**Neutral**"), report.index("Strong Positive"))

    def test_the_latest_tool_result_wins_on_a_re_entered_node(self):
        stale = _evidence()
        fresh = _evidence(company_name="Apple Inc. (fresh)")
        messages = [
            HumanMessage(content="go"),
            ToolMessage(content=json.dumps(stale.to_dict()), name=EVIDENCE_TOOL,
                        tool_call_id="a"),
            ToolMessage(content=json.dumps(fresh.to_dict()), name=EVIDENCE_TOOL,
                        tool_call_id="b"),
        ]
        report = create_earnings_analyst(FakeLLM())(
            _state(messages=messages)
        )["earnings_report"]
        self.assertIn("Apple Inc. (fresh)", report)


if __name__ == "__main__":
    unittest.main()

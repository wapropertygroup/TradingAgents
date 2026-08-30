"""Every downstream consumer must receive every analyst report.

The filename says "seven" for history; there are eight specialist reports now.
The list is derived from ``ANALYST_NODE_SPECS`` rather than written out, so
adding a ninth analyst without wiring it downstream fails here instead of
producing a report that is written to disk and then quietly ignored by every
agent that makes a decision.

The Research Manager, Trader and Portfolio Manager are covered too, and for a
different reason from the debaters. They sit behind a debate summary, a research
plan and a risk debate, each of which is lossy — so a signal that survives only
by being quoted at every hop is gone by the time it matters. They must read the
report off the state directly.
"""

import unittest
from types import SimpleNamespace

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.trader.trader import create_trader
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS

#: Derived from the graph's own registry so it cannot drift behind it.
REPORT_KEYS = sorted({spec.report_key for spec in ANALYST_NODE_SPECS.values()})

SENTINELS = {key: f"SENTINEL_{key.upper()}" for key in REPORT_KEYS}

#: Consumers that debate the analysts' evidence.
DEBATE_CONSUMERS = (
    create_bull_researcher,
    create_bear_researcher,
    create_aggressive_debator,
    create_conservative_debator,
    create_neutral_debator,
)

#: Consumers that decide. These read reports off the state directly.
DECISION_CONSUMERS = (
    create_research_manager,
    create_trader,
    create_portfolio_manager,
)


class CaptureLLM:
    """Records every prompt. Has no ``with_structured_output``, so the
    structured agents fall back to free text and their prompts land here too."""

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(str(prompt))
        return SimpleNamespace(content="ok")


def _state():
    return {
        **SENTINELS,
        "company_of_interest": "600519",
        "asset_type": "stock",
        "trade_date": "2026-08-25",
        "instrument_context": "A-share 600519",
        "trader_investment_plan": "HOLD",
        "investment_plan": "PLAN",
        "past_context": "",
        "investment_debate_state": {
            "history": "", "bull_history": "", "bear_history": "",
            "current_response": "", "judge_decision": "", "count": 0,
        },
        "risk_debate_state": {
            "history": "", "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "latest_speaker": "",
            "current_aggressive_response": "", "current_conservative_response": "",
            "current_neutral_response": "", "judge_decision": "", "count": 0,
        },
    }


class EightReportDownstreamTests(unittest.TestCase):
    def test_there_are_eight_specialist_reports(self):
        self.assertEqual(len(REPORT_KEYS), 8, REPORT_KEYS)
        self.assertIn("earnings_report", REPORT_KEYS)

    def test_debate_consumers_receive_every_report(self):
        for factory in DEBATE_CONSUMERS:
            for key, sentinel in SENTINELS.items():
                with self.subTest(factory=factory.__name__, report=key):
                    llm = CaptureLLM()
                    factory(llm)(_state())
                    self.assertIn(sentinel, " ".join(llm.prompts))

    def test_the_earnings_report_reaches_the_deciding_agents_directly(self):
        """Not relayed through a summary: the RM plan, trader proposal and risk
        debate are each lossy, and estimate revision direction is exactly the
        kind of dated, countable detail a summary drops first."""
        for factory in DECISION_CONSUMERS:
            with self.subTest(factory=factory.__name__):
                llm = CaptureLLM()
                factory(llm)(_state())
                self.assertIn(
                    SENTINELS["earnings_report"], " ".join(llm.prompts)
                )

    def test_an_absent_earnings_report_does_not_break_a_deciding_agent(self):
        """The default four-analyst selection never populates the key."""
        for factory in DECISION_CONSUMERS + DEBATE_CONSUMERS:
            with self.subTest(factory=factory.__name__):
                state = _state()
                del state["earnings_report"]
                llm = CaptureLLM()
                factory(llm)(state)
                self.assertTrue(llm.prompts)

    def test_an_empty_earnings_report_adds_no_block_to_the_deciding_agents(self):
        """An empty section header would invite comment on nothing."""
        for factory in DECISION_CONSUMERS:
            with self.subTest(factory=factory.__name__):
                state = _state()
                state["earnings_report"] = ""
                llm = CaptureLLM()
                factory(llm)(state)
                self.assertNotIn("Earnings & estimate-revision evidence", " ".join(llm.prompts))

    def test_the_deciding_agents_are_told_not_to_resolve_an_admitted_gap(self):
        """The one thing a decision must not do is convert a gap into a verdict."""
        for factory in DECISION_CONSUMERS:
            with self.subTest(factory=factory.__name__):
                llm = CaptureLLM()
                factory(llm)(_state())
                text = " ".join(llm.prompts)
                self.assertIn("Insufficient Data", text)
                self.assertIn("not", text.lower())
                self.assertTrue(
                    "neutral" in text.lower(),
                    "must warn that an unavailable field is not a neutral reading",
                )

    def test_the_bear_is_told_to_interrogate_coverage_and_confidence(self):
        llm = CaptureLLM()
        create_bear_researcher(llm)(_state())
        text = " ".join(llm.prompts)
        self.assertIn("coverage", text)
        self.assertIn("momentum band actually rests on", text)


if __name__ == "__main__":
    unittest.main()

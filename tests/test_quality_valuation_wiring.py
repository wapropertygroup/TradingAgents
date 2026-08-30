"""Wiring test: every node downstream of the analysts must see quality/valuation
evidence directly from state, not only through the (lossy) debate summary.

This is the regression guard for a specific, real risk found while building
this feature: ``bull_researcher.py``, ``bear_researcher.py``, the three risk
debators, ``research_manager.py``, ``portfolio_manager.py`` and ``trader.py``
all hardcode ``state["fundamentals_report"]`` directly into their prompts
(unlike ``portfolio_context``/``market_context``, which go through a shared
``get_..._block()`` helper). Splitting ``fundamentals`` into ``quality`` +
``valuation`` without updating all eight would have silently blinded the
debate/decision layers to this evidence while the analyst reports themselves
still rendered fine -- a regression invisible without reading the prompts.

Every node here is exercised with a full state dict and a minimal fake LLM
(free-text only, no structured-output schema needed) and the test asserts
only that quality/valuation content actually reaches the assembled prompt.
"""

import unittest

from langchain_core.messages import AIMessage

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.trader.trader import create_trader

QUALITY_MARKER = "QUALITY-REPORT-MARKER-High-Quality-tier-0.583"
VALUATION_MARKER = "VALUATION-REPORT-MARKER-Extreme-Premium-tier"


class FakeLLM:
    """Free-text only: every structured-output bind fails, forcing fallback.

    Sufficient for this test, which only checks what reaches the prompt --
    not what a real model would say back.
    """

    def __init__(self):
        self.prompts = []

    def with_structured_output(self, _schema):
        raise NotImplementedError("this fake never supports structured output")

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content="FREE TEXT RESPONSE")

    def all_prompt_text(self):
        parts = []
        for prompt in self.prompts:
            if isinstance(prompt, list):
                for m in prompt:
                    if isinstance(m, dict):
                        parts.append(str(m.get("content", "")))
                    else:
                        parts.append(str(getattr(m, "content", m)))
            else:
                parts.append(str(prompt))
        return "\n".join(parts)


def _full_state(**overrides):
    state = {
        "company_of_interest": "AAPL",
        "trade_date": "2026-08-30",
        "asset_type": "stock",
        "instrument_context": "The instrument to analyze is `AAPL`.",
        "market_report": "market ok",
        "sentiment_report": "sentiment ok",
        "news_report": "news ok",
        "fundamentals_report": "",
        "earnings_report": "",
        "quality_report": QUALITY_MARKER,
        "valuation_report": VALUATION_MARKER,
        "policy_report": "",
        "hot_money_report": "",
        "lockup_report": "",
        "portfolio_context": "",
        "portfolio_data": {},
        "market_context": "",
        "relative_strength_context": "",
        "risk_gate": {},
        "past_context": "",
        "investment_plan": "Buy, moderate conviction.",
        "trader_investment_plan": "Buy 3% of the portfolio.",
        "investment_debate_state": {
            "bull_history": "", "bear_history": "", "history": "",
            "current_response": "", "judge_decision": "", "count": 0,
        },
        "risk_debate_state": {
            "aggressive_history": "", "conservative_history": "", "neutral_history": "",
            "history": "", "latest_speaker": "",
            "current_aggressive_response": "", "current_conservative_response": "",
            "current_neutral_response": "", "judge_decision": "", "count": 0,
        },
    }
    state.update(overrides)
    return state


class DebateAndRiskNodesSeeQualityAndValuation(unittest.TestCase):
    """Plain llm.invoke() nodes: bull, bear, and the three risk debators."""

    def _assert_markers_reach_the_prompt(self, factory):
        llm = FakeLLM()
        node = factory(llm)
        node(_full_state())  # must not raise KeyError
        text = llm.all_prompt_text()
        self.assertIn(QUALITY_MARKER, text)
        self.assertIn(VALUATION_MARKER, text)

    def test_bull_researcher(self):
        self._assert_markers_reach_the_prompt(create_bull_researcher)

    def test_bear_researcher(self):
        self._assert_markers_reach_the_prompt(create_bear_researcher)

    def test_aggressive_debator(self):
        self._assert_markers_reach_the_prompt(create_aggressive_debator)

    def test_conservative_debator(self):
        self._assert_markers_reach_the_prompt(create_conservative_debator)

    def test_neutral_debator(self):
        self._assert_markers_reach_the_prompt(create_neutral_debator)

    def test_absent_reports_do_not_crash_the_prompt(self):
        """Empty strings (the real default-state value) must not raise either."""
        for factory in (create_bull_researcher, create_bear_researcher,
                       create_aggressive_debator, create_conservative_debator,
                       create_neutral_debator):
            with self.subTest(factory=factory.__name__):
                llm = FakeLLM()
                factory(llm)(_full_state(quality_report="", valuation_report=""))


class StructuredNodesSeeQualityAndValuation(unittest.TestCase):
    """Nodes that bind structured output but fall back to free text here."""

    def test_research_manager(self):
        llm = FakeLLM()
        create_research_manager(llm)(_full_state())
        text = llm.all_prompt_text()
        self.assertIn(QUALITY_MARKER, text)
        self.assertIn(VALUATION_MARKER, text)
        self.assertIn("Reconcile it explicitly", text)  # quality_block/valuation_block present

    def test_portfolio_manager(self):
        llm = FakeLLM()
        create_portfolio_manager(llm)(_full_state())
        text = llm.all_prompt_text()
        self.assertIn(QUALITY_MARKER, text)
        self.assertIn(VALUATION_MARKER, text)

    def test_trader(self):
        llm = FakeLLM()
        create_trader(llm)(_full_state())
        text = llm.all_prompt_text()
        self.assertIn(QUALITY_MARKER, text)
        self.assertIn(VALUATION_MARKER, text)

    def test_absent_reports_do_not_crash_any_structured_node(self):
        empty_state = _full_state(quality_report="", valuation_report="")
        create_research_manager(FakeLLM())(empty_state)
        create_portfolio_manager(FakeLLM())(empty_state)
        create_trader(FakeLLM())(empty_state)


if __name__ == "__main__":
    unittest.main()

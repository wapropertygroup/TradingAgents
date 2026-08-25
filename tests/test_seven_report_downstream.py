"""Every downstream debater must receive all seven analyst reports."""

import unittest
from types import SimpleNamespace

from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator


SENTINELS = {
    "market_report": "SENTINEL_MARKET",
    "sentiment_report": "SENTINEL_SENTIMENT",
    "news_report": "SENTINEL_NEWS",
    "fundamentals_report": "SENTINEL_FUNDAMENTALS",
    "policy_report": "SENTINEL_POLICY",
    "hot_money_report": "SENTINEL_HOT_MONEY",
    "lockup_report": "SENTINEL_LOCKUP",
}


class CaptureLLM:
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


class SevenReportDownstreamTests(unittest.TestCase):
    def test_bull_bear_and_all_risk_debaters_receive_all_reports(self):
        factories = (
            create_bull_researcher,
            create_bear_researcher,
            create_aggressive_debator,
            create_conservative_debator,
            create_neutral_debator,
        )
        for factory in factories:
            with self.subTest(factory=factory.__name__):
                llm = CaptureLLM()
                factory(llm)(_state())
                prompt = llm.prompts[-1]
                for sentinel in SENTINELS.values():
                    self.assertIn(sentinel, prompt)


if __name__ == "__main__":
    unittest.main()

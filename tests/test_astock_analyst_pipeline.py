"""Seven-role A-share graph contracts."""

import unittest

from tradingagents.agents.analysts.hot_money_tracker import HOT_MONEY_TOOLS
from tradingagents.agents.analysts.lockup_watcher import LOCKUP_TOOLS
from tradingagents.agents.analysts.policy_analyst import POLICY_TOOLS
from tradingagents.graph.propagation import Propagator


def _names(tools):
    return [tool.name for tool in tools]


class AStockAnalystPipelineTests(unittest.TestCase):
    def test_policy_minimum_tool_contract(self):
        self.assertEqual(_names(POLICY_TOOLS), ["get_news", "get_global_news"])

    def test_hot_money_contains_approved_minimum_and_dedicated_tools(self):
        names = _names(HOT_MONEY_TOOLS)
        for required in ("get_stock_data", "get_news", "get_insider_transactions"):
            self.assertIn(required, names)
        self.assertIn("get_dragon_tiger_board", names)
        self.assertIn("get_fund_flow", names)

    def test_lockup_contains_approved_minimum_and_calendar(self):
        names = _names(LOCKUP_TOOLS)
        for required in ("get_insider_transactions", "get_news", "get_fundamentals"):
            self.assertIn(required, names)
        self.assertIn("get_lockup_expiry", names)

    def test_initial_state_has_all_specialist_reports(self):
        state = Propagator().create_initial_state("600519", "2026-08-25")
        self.assertEqual(state["policy_report"], "")
        self.assertEqual(state["hot_money_report"], "")
        self.assertEqual(state["lockup_report"], "")


if __name__ == "__main__":
    unittest.main()

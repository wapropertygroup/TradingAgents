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
        """Derived from the graph registry, so a new analyst that forgets to
        initialise its report key fails here rather than raising a KeyError
        mid-run in whichever downstream agent reads it first."""
        from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS

        state = Propagator().create_initial_state("600519", "2026-08-25")
        for spec in ANALYST_NODE_SPECS.values():
            with self.subTest(report=spec.report_key):
                self.assertEqual(state[spec.report_key], "")

    def test_earnings_report_starts_empty(self):
        state = Propagator().create_initial_state("600519", "2026-08-25")
        self.assertEqual(state["earnings_report"], "")

    def test_earnings_tool_contract(self):
        """Both tools must be bound: the analyst calls them deterministically,
        so an unregistered one fails and the report describes its own plumbing
        as missing evidence."""
        from tradingagents.agents.analysts.earnings_analyst import (
            COMMENTARY_TOOL,
            EVIDENCE_TOOL,
        )
        from tradingagents.agents.utils.earnings_data_tools import (
            get_earnings_commentary,
            get_earnings_evidence,
        )

        self.assertEqual(get_earnings_evidence.name, EVIDENCE_TOOL)
        self.assertEqual(get_earnings_commentary.name, COMMENTARY_TOOL)


if __name__ == "__main__":
    unittest.main()

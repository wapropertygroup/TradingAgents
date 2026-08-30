"""Full-state JSON logging must preserve every specialist report.

The audit log used to be built by subscripting exactly four report keys, which
had two consequences. An unselected analyst raised ``KeyError`` *after* the run
finished and the API spend was gone. And the three A-share reports plus this
feature's earnings report were silently dropped even when they had run — the file
read as a complete record of the analysis while omitting whichever specialists
were selected, which is worse than having no audit trail.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from tradingagents.graph.trading_graph import TradingAgentsGraph

SPECIALIST_REPORTS = (
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "earnings_report",
    "policy_report",
    "hot_money_report",
    "lockup_report",
)


def _final_state(**overrides):
    state = {
        "company_of_interest": "AAPL",
        "trade_date": "2026-08-30",
        "investment_debate_state": {
            "bull_history": "BULL", "bear_history": "BEAR", "history": "HISTORY",
            "current_response": "CURRENT", "judge_decision": "JUDGE",
        },
        "trader_investment_plan": "TRADER_PLAN",
        "risk_debate_state": {
            "aggressive_history": "AGG", "conservative_history": "CON",
            "neutral_history": "NEU", "history": "RISK_HISTORY",
            "judge_decision": "PM_DECISION",
        },
        "investment_plan": "INVESTMENT_PLAN",
        "final_trade_decision": "FINAL",
    }
    for key in SPECIALIST_REPORTS:
        state[key] = f"CONTENT_{key.upper()}"
    state.update(overrides)
    return state


class FullStateLoggingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.results_dir = Path(self._tmp.name)
        # Build the object without touching LLM providers or the network.
        self.graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
        self.graph.config = {"results_dir": str(self.results_dir)}
        self.graph.log_states_dict = {}
        self.graph.ticker = "AAPL"

    def tearDown(self):
        self._tmp.cleanup()

    def _log(self, state):
        self.graph._log_state("2026-08-30", state)
        path = (
            self.results_dir / "AAPL" / "TradingAgentsStrategy_logs"
            / "full_states_log_2026-08-30.json"
        )
        self.assertTrue(path.exists(), "the audit log must be written")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_every_specialist_report_survives_to_disk(self):
        logged = self._log(_final_state())
        for key in SPECIALIST_REPORTS:
            with self.subTest(report=key):
                self.assertIn(key, logged)
                self.assertEqual(logged[key], f"CONTENT_{key.upper()}")

    def test_the_earnings_report_is_logged(self):
        self.assertEqual(
            self._log(_final_state())["earnings_report"], "CONTENT_EARNINGS_REPORT"
        )

    def test_the_three_a_share_reports_are_logged(self):
        logged = self._log(_final_state())
        for key in ("policy_report", "hot_money_report", "lockup_report"):
            with self.subTest(report=key):
                self.assertEqual(logged[key], f"CONTENT_{key.upper()}")

    def test_an_unselected_analyst_does_not_raise_after_the_run_completes(self):
        """A KeyError here discards a finished analysis and its API spend."""
        state = _final_state()
        for key in ("earnings_report", "policy_report", "hot_money_report",
                    "lockup_report", "fundamentals_report"):
            del state[key]
        logged = self._log(state)
        for key in ("earnings_report", "policy_report", "fundamentals_report"):
            with self.subTest(report=key):
                self.assertEqual(logged[key], "")

    def test_a_bare_state_with_only_the_required_keys_still_logs(self):
        state = _final_state()
        for key in SPECIALIST_REPORTS:
            del state[key]
        logged = self._log(state)
        self.assertEqual({logged[key] for key in SPECIALIST_REPORTS}, {""})

    def test_the_decision_chain_is_preserved(self):
        logged = self._log(_final_state())
        self.assertEqual(logged["investment_plan"], "INVESTMENT_PLAN")
        self.assertEqual(logged["trader_investment_decision"], "TRADER_PLAN")
        self.assertEqual(logged["final_trade_decision"], "FINAL")
        self.assertEqual(logged["investment_debate_state"]["bull_history"], "BULL")
        self.assertEqual(logged["risk_debate_state"]["judge_decision"], "PM_DECISION")

    def test_the_log_is_valid_utf8_json_for_non_ascii_report_content(self):
        state = _final_state(policy_report="政策收紧，估值承压")
        logged = self._log(state)
        self.assertEqual(logged["policy_report"], "政策收紧，估值承压")

    def test_a_ticker_that_would_escape_the_results_directory_is_rejected(self):
        self.graph.ticker = "../../etc/passwd"
        with self.assertRaises(ValueError):
            self.graph._log_state("2026-08-30", _final_state())


class ReportTreeParityTests(unittest.TestCase):
    """The markdown tree and the JSON log must agree on what ran."""

    def test_earnings_is_written_and_titled_in_the_consolidated_report(self):
        from tradingagents.reporting import write_report_tree

        with TemporaryDirectory() as tmp:
            path = write_report_tree(_final_state(), "AAPL", Path(tmp) / "reports")
            earnings_md = Path(tmp) / "reports" / "1_analysts" / "earnings.md"
            self.assertTrue(earnings_md.exists())
            self.assertEqual(earnings_md.read_text(encoding="utf-8"), "CONTENT_EARNINGS_REPORT")
            consolidated = path.read_text(encoding="utf-8")
            self.assertIn("### Earnings Analyst", consolidated)
            self.assertIn("CONTENT_EARNINGS_REPORT", consolidated)

    def test_an_absent_earnings_report_writes_no_file(self):
        from tradingagents.reporting import write_report_tree

        state = _final_state()
        del state["earnings_report"]
        with TemporaryDirectory() as tmp:
            path = write_report_tree(state, "AAPL", Path(tmp) / "reports")
            self.assertFalse((Path(tmp) / "reports" / "1_analysts" / "earnings.md").exists())
            self.assertNotIn("### Earnings Analyst", path.read_text(encoding="utf-8"))


class ToolNodeRegistrationTests(unittest.TestCase):
    def test_the_earnings_tool_node_can_execute_both_tools_the_analyst_calls(self):
        """The analyst calls both deterministically; an unregistered tool fails.

        The failure is quiet in the worst way: the report would describe its own
        plumbing as missing evidence.
        """
        from tradingagents.agents.analysts.earnings_analyst import (
            COMMENTARY_TOOL,
            EVIDENCE_TOOL,
        )

        graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
        nodes = graph._create_tool_nodes()
        self.assertIn("earnings", nodes)
        registered = {tool.name for tool in nodes["earnings"].tools_by_name.values()}
        self.assertEqual(registered, {EVIDENCE_TOOL, COMMENTARY_TOOL})

    def test_every_analyst_spec_has_a_matching_tool_node(self):
        from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS

        graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
        nodes = graph._create_tool_nodes()
        for key in ANALYST_NODE_SPECS:
            with self.subTest(analyst=key):
                self.assertIn(key, nodes)


class CheckpointSignatureTests(unittest.TestCase):
    def test_changing_the_earnings_selection_changes_the_graph_signature(self):
        """A resume under a different analyst set must not continue the old graph."""
        def signature(selected):
            graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
            graph.selected_analysts = tuple(selected)
            graph.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
            return graph._run_signature("stock")

        without = signature(["market", "social", "news", "fundamentals"])
        with_earnings = signature(["market", "social", "news", "fundamentals", "earnings"])
        self.assertNotEqual(without, with_earnings)
        self.assertIn("earnings", with_earnings)
        self.assertNotIn("earnings", without)


if __name__ == "__main__":
    unittest.main()

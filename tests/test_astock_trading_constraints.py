"""A-share market mechanics are conditional, not leaked to US symbols."""

import unittest

from tradingagents.agents.managers.portfolio_manager import _A_SHARE_FINAL_CONSTRAINTS
from tradingagents.agents.trader.trader import _A_SHARE_CONSTRAINTS
from tradingagents.dataflows.a_stock import is_a_share


class AStockTradingConstraintTests(unittest.TestCase):
    def test_a_share_forms_are_detected(self):
        for ticker in ("600519", "SH600519", "600519.SS", "000001.SZ", "BJ920002"):
            with self.subTest(ticker=ticker):
                self.assertTrue(is_a_share(ticker))

    def test_us_ticker_is_not_a_share(self):
        self.assertFalse(is_a_share("AAPL"))

    def test_constraint_text_covers_execution_mechanics(self):
        combined = _A_SHARE_CONSTRAINTS + _A_SHARE_FINAL_CONSTRAINTS
        for term in ("T+1", "100-share", "price limit", "suspension"):
            self.assertIn(term, combined)

    def test_the_board_rules_are_current(self):
        """As TradingAgents-astock corrected them, as of the 2026-07-06 change:
        the Beijing Stock Exchange has its own band, listing-day exception and
        lot rule, and ST/*ST no longer narrows the band."""
        for raw in (_A_SHARE_CONSTRAINTS, _A_SHARE_FINAL_CONSTRAINTS):
            text = " ".join(raw.split())        # the prompt's line breaks carry nothing
            with self.subTest(text=text[:40]):
                self.assertIn("Beijing Stock Exchange ±30%", text)
                self.assertIn("first 5 trading days", text)
                self.assertIn("first day only" if raw is _A_SHARE_FINAL_CONSTRAINTS
                              else "its first day only", text)
                self.assertRegex(text, r"200 shares minimum on STAR|on STAR 200 shares minimum")
                self.assertNotIn("5% for ST", text)
                self.assertNotIn("5%/10%/20%", text)


if __name__ == "__main__":
    unittest.main()

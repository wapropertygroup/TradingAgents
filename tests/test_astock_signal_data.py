"""Deterministic tests for A-share specialist source adapters."""

import unittest
from unittest.mock import patch

from tradingagents.dataflows import a_stock


class AStockSignalDataTests(unittest.TestCase):
    def test_lockup_calendar_formats_upcoming_supply(self):
        rows = [{
            "FREE_DATE": "2026-09-01", "LIMITED_STOCK_TYPE": "定增限售",
            "FREE_SHARES_NUM": 1000000, "FREE_RATIO": 3.2,
        }]
        with patch.object(a_stock, "_datacenter", return_value=rows):
            text = a_stock.get_lockup_expiry("600519", "2026-08-25")
        self.assertIn("2026-09-01", text)
        self.assertIn("定增限售", text)
        self.assertIn("3.2", text)

    def test_dragon_tiger_no_appearance_is_explicit(self):
        with patch.object(a_stock, "_datacenter", return_value=[]):
            text = a_stock.get_dragon_tiger_board("000001", "2026-08-25")
        self.assertIn("未查到上榜记录", text)

    def test_profit_forecast_warns_for_historical_snapshot(self):
        with patch.object(a_stock, "_eps_forecast_ths", return_value="| 年 | EPS |"):
            text = a_stock.get_profit_forecast("600519", "2020-01-02")
        self.assertIn("未来函数警告", text)
        self.assertIn("同花顺", text)

    def test_fund_flow_filters_rows_after_analysis_date(self):
        payloads = [
            {"data": {"klines": []}},
            {"data": {"klines": [
                "2026-08-24,1,2,3,4,5", "2026-08-26,9,9,9,9,9",
            ]}},
        ]
        with patch.object(a_stock, "_historical_notice", return_value=""), \
             patch.object(a_stock, "_em_json", side_effect=payloads):
            text = a_stock.get_fund_flow("600519", "2026-08-25")
        self.assertIn("2026-08-24", text)
        self.assertNotIn("2026-08-26", text)


if __name__ == "__main__":
    unittest.main()

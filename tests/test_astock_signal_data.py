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


class AStockPortedFixTests(unittest.TestCase):
    """Fixes ported from TradingAgents-astock (eaf3876, 5789834, a13f336)."""

    _SNAPSHOT = {"f58": "贵州茅台", "f43": 150000, "f164": 2500}

    def test_fundamentals_for_a_past_date_warn_and_claim_no_vintage(self):
        with patch.object(a_stock, "_snapshot", return_value=self._SNAPSHOT), \
             patch.object(a_stock, "_eps_forecast_ths", return_value=""):
            past = a_stock.get_fundamentals("600519", "2020-01-02")
            today = a_stock.get_fundamentals("600519", None)
        self.assertIn("未来函数警告", past)
        self.assertNotIn("as of 2020-01-02", past)
        self.assertIn("当前快照", past)
        self.assertNotIn("未来函数警告", today)

    def test_a_past_fund_flow_window_reaches_back_to_the_analysis_date(self):
        asked = []

        def em_json(url, params):
            asked.append(params.get("lmt"))
            return {"data": {"klines": ["2019-12-30,1,2,3,4,5", "2020-01-03,9,9,9,9,9"]}}

        with patch.object(a_stock, "_em_json", side_effect=em_json):
            text = a_stock.get_fund_flow("600519", "2020-01-02")
        self.assertEqual(asked, [500])          # capped; years back from today
        self.assertIn("2019-12-30", text)
        self.assertNotIn("2020-01-03", text)

    def test_a_date_beyond_the_window_is_out_of_reach_not_no_flow(self):
        with patch.object(a_stock, "_em_json",
                          return_value={"data": {"klines": ["2024-06-03,1,2,3,4,5"]}}):
            text = a_stock.get_fund_flow("600519", "2020-01-02")
        self.assertTrue(text.startswith("DATA_UNAVAILABLE"), text)
        self.assertIn("超出可回溯范围", text)
        self.assertNotIn("NO_DATA_AVAILABLE", text)

    def test_a_blocked_concept_block_request_is_the_vendor_declining(self):
        import json

        from tradingagents.dataflows.errors import VendorUnavailableError

        blocked = json.dumps({"ResultCode": 0, "Result": {
            "code": 403, "isCaptchaEnabled": True, "msg": "hit risk"}})
        with patch.object(a_stock, "_http", return_value=blocked), \
             self.assertRaises(VendorUnavailableError):
            a_stock.get_concept_blocks("600519")


if __name__ == "__main__":
    unittest.main()

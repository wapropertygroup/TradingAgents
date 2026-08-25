"""mootdx is optional and must not prevent HTTP fallback."""

import os
import unittest
from unittest.mock import patch

import pandas as pd

from tradingagents.dataflows import a_stock


def _frame():
    return pd.DataFrame([{
        "Date": "2026-08-25", "Open": 1, "High": 2, "Low": 1,
        "Close": 2, "Volume": 100,
    }])


class MootdxCompatibilityTests(unittest.TestCase):
    def test_disabled_mootdx_falls_back_to_eastmoney(self):
        with patch.dict(os.environ, {"TRADINGAGENTS_MOOTDX_ENABLED": "0"}), \
             patch.object(a_stock, "_fetch_kline_em", return_value=_frame()):
            result = a_stock.load_ohlcv("600519", "2026-08-25")
        self.assertEqual(result.attrs["source"], "eastmoney")

    def test_missing_mootdx_import_falls_back_without_env_var(self):
        """The production case: the package was never installed at all."""
        env = {k: v for k, v in os.environ.items()
               if k != "TRADINGAGENTS_MOOTDX_ENABLED"}
        with patch.dict(os.environ, env, clear=True), \
             patch.object(a_stock, "_fetch_kline_mootdx",
                          side_effect=ImportError("No module named 'mootdx'")), \
             patch.object(a_stock, "_fetch_kline_em", return_value=_frame()):
            result = a_stock.load_ohlcv("600519", "2026-08-25")
        self.assertEqual(result.attrs["source"], "eastmoney")

    def test_eastmoney_rate_limit_reaches_sina(self):
        with patch.dict(os.environ, {"TRADINGAGENTS_MOOTDX_ENABLED": "0"}), \
             patch.object(a_stock, "_fetch_kline_em",
                          side_effect=a_stock.VendorRateLimitError("429")), \
             patch.object(a_stock, "_fetch_kline_sina", return_value=_frame()):
            result = a_stock.load_ohlcv("600519", "2026-08-25")
        self.assertEqual(result.attrs["source"], "sina")

    def test_reshaped_eastmoney_payload_reaches_sina(self):
        """东财 breaks by changing its field list, not by going down.

        Truncated kline fields make the row parser skip every line, so the frame
        arrives with no columns. That used to surface as ``KeyError: 'Date'``,
        which escaped the fallback chain and aborted the run instead of trying
        新浪.
        """
        reshaped = {"data": {"name": "贵州茅台", "klines": ["2026-08-25,1,2"]}}
        with patch.dict(os.environ, {"TRADINGAGENTS_MOOTDX_ENABLED": "0"}), \
             patch.object(a_stock, "_em_json", return_value=reshaped), \
             patch.object(a_stock, "_fetch_kline_sina", return_value=_frame()):
            result = a_stock.load_ohlcv("600519", "2026-08-25")
        self.assertEqual(result.attrs["source"], "sina")

    def test_columnless_frame_is_a_vendor_error_not_keyerror(self):
        with self.assertRaises(a_stock.NoMarketDataError):
            a_stock._clean_ohlcv(pd.DataFrame([]))


if __name__ == "__main__":
    unittest.main()

"""Vendor routing for the eight A-share signal tools.

The three A-share analysts reached production able to *resolve* their tools but
not to *route* them: a 东财 throttle on the single signal vendor exhausted the
chain, and route_to_vendor fell through to ``RuntimeError: No available vendor for
'get_fund_flow'``. That killed the tool node and with it the whole run, discarding
six other analysts' completed work -- and it named a registration bug that did not
exist instead of the throttle that did.

Asserting that a tool object exists was not enough. These tests assert the call
actually round-trips.
"""

import copy
import unittest
from unittest.mock import patch

import tradingagents.default_config as default_config
from tradingagents.dataflows import interface as I
from tradingagents.dataflows.a_stock import NoMarketDataError, VendorRateLimitError
from tradingagents.dataflows.config import set_config


# The eight tools signal_data_tools.py exposes to the policy, hot-money and
# lock-up analysts.
SIGNAL_METHODS = (
    "get_profit_forecast",
    "get_hot_stocks",
    "get_northbound_flow",
    "get_concept_blocks",
    "get_fund_flow",
    "get_dragon_tiger_board",
    "get_lockup_expiry",
    "get_industry_comparison",
)


def _raising(exc):
    def _impl(*_args, **_kwargs):
        raise exc
    return _impl


class SignalToolRegistrationTests(unittest.TestCase):
    def setUp(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def test_every_signal_tool_is_fully_registered(self):
        """A tool needs a category *and* a vendor impl; either alone is a dead end."""
        for method in SIGNAL_METHODS:
            with self.subTest(method=method):
                self.assertEqual(I.get_category_for_method(method), "signal_data")
                self.assertIn(method, I.VENDOR_METHODS)
                self.assertIn("a_stock", I.VENDOR_METHODS[method])

    def test_configured_signal_vendor_is_actually_available(self):
        """The configured chain must intersect the registered vendors.

        default_config notes that registering in VENDOR_METHODS is not sufficient
        on its own -- an explicit chain IS the whole chain -- so a typo in either
        place makes every one of these tools raise ValueError at run time.
        """
        for method in SIGNAL_METHODS:
            with self.subTest(method=method):
                configured = I.get_vendor("signal_data", method)
                chain = [v.strip() for v in configured.split(",") if v.strip()]
                self.assertTrue(
                    set(chain) & set(I.VENDOR_METHODS[method]),
                    f"configured {chain} matches none of "
                    f"{list(I.VENDOR_METHODS[method])}",
                )


class SignalToolThrottleTests(unittest.TestCase):
    """signal_data has one vendor, so a throttle exhausts its chain immediately."""

    def setUp(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def test_throttled_signal_vendor_degrades_instead_of_raising(self):
        for method in SIGNAL_METHODS:
            with self.subTest(method=method):
                with patch.dict(
                    I.VENDOR_METHODS[method],
                    {"a_stock": _raising(VendorRateLimitError("东财 429"))},
                ):
                    result = I.route_to_vendor(method, "600519", "2026-08-25")
                self.assertIsInstance(result, str)
                self.assertIn("DATA_UNAVAILABLE", result)
                # The analyst must be told not to invent numbers in place of the
                # data it could not read.
                self.assertIn("do not fabricate", result.lower())

    def test_throttle_is_never_reported_as_a_missing_vendor(self):
        """The exact production regression, pinned by its message."""
        with patch.dict(
            I.VENDOR_METHODS["get_fund_flow"],
            {"a_stock": _raising(VendorRateLimitError("东财 429"))},
        ):
            result = I.route_to_vendor("get_fund_flow", "600519", "2026-08-25", False)
        self.assertNotIn("No available vendor", result)

    def test_genuine_no_data_still_takes_precedence(self):
        with patch.dict(
            I.VENDOR_METHODS["get_fund_flow"],
            {"a_stock": _raising(NoMarketDataError("ZZZZ", None, "not an A-share code"))},
        ):
            result = I.route_to_vendor("get_fund_flow", "ZZZZ", "2026-08-25", False)
        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertIn("not an A-share code", result)


class CoreCategoryThrottleTests(unittest.TestCase):
    """Core data must stay loud: degrading prices would hide a broken primary."""

    def setUp(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def test_fully_throttled_core_category_names_the_throttle(self):
        throttled = {
            vendor: _raising(VendorRateLimitError(f"{vendor} 429"))
            for vendor in I.VENDOR_METHODS["get_stock_data"]
        }
        with patch.dict(I.VENDOR_METHODS["get_stock_data"], throttled):
            with self.assertRaises(VendorRateLimitError):
                I.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-08-25")

    def test_core_category_is_not_optional(self):
        self.assertNotIn("core_stock_apis", I.OPTIONAL_CATEGORIES)
        self.assertNotIn("news_data", I.OPTIONAL_CATEGORIES)
        self.assertIn("signal_data", I.OPTIONAL_CATEGORIES)


if __name__ == "__main__":
    unittest.main()

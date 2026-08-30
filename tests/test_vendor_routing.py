"""Vendor router must respect the configured chain and never silently hide a
broken primary.

Regressions for #988 (explicit single-vendor config still fell back to others),
#289 (fallback ran for unchosen vendors), and #989 (serious primary failures
were swallowed without a trace).
"""
import copy
import unittest
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.symbol_utils import NoMarketDataError


def _reset_config():
    # Hard reset: set_config() merges, so empty DEFAULT dicts (e.g. tool_vendors)
    # don't clear keys leaked by other tests. Replace the global outright.
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


def _no_data(symbol, *a, **k):
    raise NoMarketDataError(symbol, symbol, "no rows")


def _returns(value):
    def impl(symbol, *a, **k):
        return value
    return impl


def _raises(exc):
    def impl(symbol, *a, **k):
        raise exc
    return impl


@pytest.mark.unit
class VendorRoutingTests(unittest.TestCase):
    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def _route(self, vendors_for_get_stock_data):
        return mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": vendors_for_get_stock_data},
            clear=False,
        )

    def test_explicit_single_vendor_does_not_fall_back(self):
        # #988: with yfinance pinned, a healthy alpha_vantage must NOT be used.
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        av = mock.Mock(side_effect=_returns("AV_DATA"))
        with self._route({"yfinance": _no_data, "alpha_vantage": av}):
            result = interface.route_to_vendor("get_stock_data", "FAKE", "2026-01-01", "2026-01-10")
        self.assertIn("NO_DATA_AVAILABLE", result)
        av.assert_not_called()  # the unchosen vendor was never tried

    def test_explicit_multi_vendor_falls_back_within_chain(self):
        # Listing both vendors opts in to ordered fallback.
        set_config({"data_vendors": {"core_stock_apis": "yfinance,alpha_vantage"}})
        with self._route({"yfinance": _no_data, "alpha_vantage": _returns("AV_DATA")}):
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(result, "AV_DATA")

    def test_primary_error_is_logged_not_masked(self):
        # #989: primary errors + fallback no-data -> NO_DATA, but the failure
        # must be visible in logs (broken primary not hidden).
        set_config({"data_vendors": {"core_stock_apis": "yfinance,alpha_vantage"}})
        with self._route({"yfinance": _raises(ValueError("boom")), "alpha_vantage": _no_data}), \
                self.assertLogs("tradingagents.dataflows.interface", level="WARNING") as cm:
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertIn("NO_DATA_AVAILABLE", result)
        joined = "\n".join(cm.output)
        self.assertIn("boom", joined)            # the real error surfaced in logs
        self.assertIn("yfinance", joined)

    def test_unknown_configured_vendor_raises(self):
        set_config({"data_vendors": {"core_stock_apis": "bogus_vendor"}})
        with self.assertRaises(ValueError) as ctx:
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertIn("bogus_vendor", str(ctx.exception))

    def test_default_sentinel_uses_all_vendors(self):
        # No explicit choice ("default") keeps the resilient full-chain behavior.
        set_config({"data_vendors": {"core_stock_apis": "default"}})
        with self._route({"yfinance": _no_data, "alpha_vantage": _returns("AV_DATA")}):
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(result, "AV_DATA")

    def _route_method(self, method, vendors):
        return mock.patch.dict(interface.VENDOR_METHODS, {method: vendors}, clear=False)

    def test_optional_category_degrades_instead_of_raising(self):
        # An optional enrichment vendor (FRED macro) that raises must NOT abort
        # the run — the router returns a sentinel so the analysis proceeds.
        set_config({"data_vendors": {"macro_data": "fred"}})
        with self._route_method(
            "get_macro_indicators", {"fred": _raises(ValueError("FRED 400: bad series"))}
        ):
            result = interface.route_to_vendor("get_macro_indicators", "cpi", "2026-01-01")
        self.assertIn("DATA_UNAVAILABLE", result)
        self.assertIn("macro_data", result)

    def test_core_category_still_raises_on_error(self):
        # A core category (single configured vendor) propagates the error so a
        # broken primary is loud, not silently degraded.
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        with self._route({"yfinance": _raises(ValueError("boom"))}), \
                self.assertRaises(ValueError):
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")


@pytest.mark.unit
class EarningsRoutingTests(unittest.TestCase):
    """The two earnings categories, and why their failure modes differ."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def _route_method(self, method, vendors):
        return mock.patch.dict(interface.VENDOR_METHODS, {method: vendors}, clear=False)

    def test_both_categories_are_registered(self):
        self.assertEqual(
            interface.get_category_for_method("get_earnings_evidence"), "earnings_data"
        )
        self.assertEqual(
            interface.get_category_for_method("get_earnings_commentary"),
            "earnings_commentary",
        )

    def test_the_default_chain_puts_yfinance_ahead_of_a_stock(self):
        """The reverse of the four core chains, and deliberately so.

        Yahoo publishes a real revision history for every venue it covers, and
        unlike the price and news paths this adapter cannot answer emptily: it
        refuses a bare 6-digit code by inspection and raises on an unknown symbol.
        """
        chain = default_config.DEFAULT_CONFIG["data_vendors"]["earnings_data"]
        self.assertEqual(chain, "yfinance,a_stock")

    def test_alpha_vantage_is_registered_but_not_in_the_default_chain(self):
        """Opt-in: no run may acquire an API-key dependency without being asked."""
        self.assertIn("alpha_vantage", interface.VENDOR_METHODS["get_earnings_evidence"])
        self.assertNotIn(
            "alpha_vantage", default_config.DEFAULT_CONFIG["data_vendors"]["earnings_data"]
        )

    def test_the_configured_chain_is_the_whole_chain(self):
        """An unlisted vendor is never tried, even when it would succeed."""
        set_config({"data_vendors": {"earnings_data": "a_stock"}})
        with self._route_method("get_earnings_evidence", {
            "yfinance": _returns("YF_JSON"),
            "a_stock": _no_data,
        }):
            result = interface.route_to_vendor("get_earnings_evidence", "AAPL", "2026-08-30")
        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertNotIn("YF_JSON", result)

    def test_a_refusing_primary_falls_through_to_the_next_configured_vendor(self):
        set_config({"data_vendors": {"earnings_data": "yfinance,a_stock"}})
        with self._route_method("get_earnings_evidence", {
            "yfinance": _no_data,
            "a_stock": _returns("THS_JSON"),
        }):
            result = interface.route_to_vendor("get_earnings_evidence", "600519", "2026-08-30")
        self.assertEqual(result, "THS_JSON")

    def test_an_exhausted_chain_returns_one_instructive_sentinel(self):
        set_config({"data_vendors": {"earnings_data": "yfinance,a_stock"}})
        with self._route_method("get_earnings_evidence", {
            "yfinance": _no_data,
            "a_stock": _no_data,
        }):
            result = interface.route_to_vendor("get_earnings_evidence", "ZZZZ", "2026-08-30")
        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertIn("Do not estimate or fabricate", result)

    def test_earnings_data_is_not_optional_so_a_broken_vendor_is_loud(self):
        """It is the analyst's core payload, like fundamentals or news.

        The routine "no earnings" and "no vintage" outcomes never reach here —
        the adapters return those as structured evidence with an explicit status.
        """
        self.assertNotIn("earnings_data", interface.OPTIONAL_CATEGORIES)
        set_config({"data_vendors": {"earnings_data": "yfinance"}})
        with self._route_method(
            "get_earnings_evidence", {"yfinance": _raises(ValueError("yahoo exploded"))}
        ), self.assertRaises(ValueError):
            interface.route_to_vendor("get_earnings_evidence", "AAPL", "2026-08-30")

    def test_earnings_commentary_is_optional_and_degrades_to_a_sentinel(self):
        """Its only vendor is premium-gated, so exhaustion is the common case.

        A missing transcript must not abort a report whose numeric evidence is
        already in hand.
        """
        self.assertIn("earnings_commentary", interface.OPTIONAL_CATEGORIES)
        set_config({"data_vendors": {"earnings_commentary": "alpha_vantage"}})
        with self._route_method(
            "get_earnings_commentary",
            {"alpha_vantage": _raises(interface.VendorNotConfiguredError("no key"))},
        ):
            result = interface.route_to_vendor("get_earnings_commentary", "AAPL", "2026-08-30")
        self.assertIn("DATA_UNAVAILABLE", result)
        self.assertIn("earnings_commentary", result)
        self.assertIn("do not fabricate", result)

    def test_an_unconfigured_commentary_vendor_makes_no_network_call(self):
        calls = []

        def record(symbol, *a, **k):
            calls.append(symbol)
            raise interface.VendorNotConfiguredError("ALPHA_VANTAGE_API_KEY is not set")

        set_config({"data_vendors": {"earnings_commentary": "alpha_vantage"}})
        with self._route_method("get_earnings_commentary", {"alpha_vantage": record}):
            interface.route_to_vendor("get_earnings_commentary", "AAPL", "2026-08-30")
        # The vendor is entered once and refuses before reaching the network; the
        # assertion here is that the router does not retry or reach elsewhere.
        self.assertEqual(len(calls), 1)

    def test_a_tool_level_override_beats_the_category_chain(self):
        set_config({
            "data_vendors": {"earnings_data": "yfinance"},
            "tool_vendors": {"get_earnings_evidence": "alpha_vantage"},
        })
        with self._route_method("get_earnings_evidence", {
            "yfinance": _returns("YF"),
            "alpha_vantage": _returns("AV"),
        }):
            self.assertEqual(
                interface.route_to_vendor("get_earnings_evidence", "AAPL", "2026-08-30"),
                "AV",
            )


if __name__ == "__main__":
    unittest.main()

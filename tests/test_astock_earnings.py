"""A-share earnings adapter: 同花顺 normalization, refusal, and honest gaps.

The forecast table's layout was verified against the live page for 600519 and
cross-checked against Yahoo's CNY consensus for the same issuer, which is what
makes normalizing it safe. These tests pin both the mapping and the refusal to
map an unrecognised layout.
"""

import json
import unittest
from unittest.mock import patch

import pandas as pd

from tradingagents.dataflows import a_stock_earnings as ase
from tradingagents.dataflows.errors import NoMarketDataError

# The live 600519 table, verbatim.
THS_TABLE = pd.DataFrame(
    {
        "年度": [2026, 2027, 2028],
        "预测机构数": [49, 48, 44],
        "最小值": [64.78, 67.23, 69.14],
        "均值": [67.85, 71.79, 75.24],
        "最大值": [77.85, 84.02, 83.83],
        "行业平均数": [8.07, 8.64, 9.03],
    }
)

AS_OF = "2026-08-30"


class AShareTestCase(unittest.TestCase):
    def setUp(self):
        from tempfile import TemporaryDirectory

        self._tmp = TemporaryDirectory()
        self._config = patch.dict(
            "tradingagents.dataflows.config._config", {"data_cache_dir": self._tmp.name}
        )
        self._config.start()

    def tearDown(self):
        self._config.stop()
        self._tmp.cleanup()

    def build(self, symbol="600519", as_of=AS_OF, frame=THS_TABLE):
        with patch.object(ase, "_eps_forecast_ths_frame", return_value=frame):
            return ase.build_earnings_evidence(symbol, as_of)


class RefusalTests(AShareTestCase):
    def test_a_non_a_share_symbol_is_refused_so_the_router_moves_on(self):
        def explode(_code):
            raise AssertionError("must refuse before fetching")

        with patch.object(ase, "_eps_forecast_ths_frame", explode):
            for symbol in ("AAPL", "SPY", "BTC-USD", "7203.T", "0700.HK"):
                with self.subTest(symbol=symbol):
                    with self.assertRaises(NoMarketDataError):
                        ase.build_earnings_evidence(symbol, AS_OF)

    def test_decorated_a_share_forms_are_accepted(self):
        for symbol in ("600519", "sh600519", "600519.SS", "600519.SH"):
            with self.subTest(symbol=symbol):
                evidence = self.build(symbol=symbol)
                self.assertEqual(evidence.canonical_symbol, "600519")

    def test_a_missing_page_is_no_coverage_with_a_reason(self):
        for frame in (None, pd.DataFrame()):
            with self.subTest(frame=type(frame).__name__):
                evidence = self.build(frame=frame)
                self.assertEqual(evidence.status, "no_coverage")
                self.assertIn("no consensus-forecast table", evidence.status_detail)


class NormalizationTests(AShareTestCase):
    def setUp(self):
        super().setUp()
        self.evidence = self.build()

    def test_explicit_fiscal_years_map_onto_relative_period_keys(self):
        self.assertEqual(sorted(self.evidence.periods), sorted(["0y", "+1y", "+2y"]))

    def test_consensus_mean_is_read_from_the_mean_column(self):
        self.assertAlmostEqual(self.evidence.periods["0y"].eps.current.value, 67.85)
        self.assertAlmostEqual(self.evidence.periods["+1y"].eps.current.value, 71.79)
        self.assertAlmostEqual(self.evidence.periods["+2y"].eps.current.value, 75.24)

    def test_the_industry_average_column_is_not_mistaken_for_this_issuer(self):
        for period in self.evidence.periods.values():
            self.assertNotAlmostEqual(period.eps.current.value, 8.07)
            self.assertNotAlmostEqual(period.eps.current.value, 8.64)

    def test_covering_institution_count_becomes_analyst_coverage(self):
        self.assertAlmostEqual(self.evidence.periods["0y"].analyst_count.value, 49)
        self.assertAlmostEqual(self.evidence.periods["+1y"].analyst_count.value, 48)

    def test_fiscal_year_end_follows_the_calendar_year_by_regulation(self):
        """沪深京 issuers must use the calendar year, so this is not a guess."""
        self.assertEqual(self.evidence.periods["0y"].period.end_date, "2026-12-31")
        self.assertEqual(self.evidence.periods["+2y"].period.end_date, "2028-12-31")
        self.assertIn("FY2026", self.evidence.periods["0y"].period.label)

    def test_currency_is_declared_so_a_figure_is_never_bare_beside_a_usd_one(self):
        self.assertEqual(self.evidence.currency, "CNY")
        self.assertEqual(self.evidence.periods["0y"].eps.current.currency, "CNY")

    def test_the_estimate_range_is_recorded_in_the_source_note(self):
        source = self.evidence.periods["0y"].eps.current.source
        self.assertIn("low 64.78", source)
        self.assertIn("high 77.85", source)

    def test_a_past_fiscal_year_row_is_dropped_rather_than_read_as_a_forecast(self):
        """A realised figure published as a forward consensus is the core risk."""
        stale = pd.concat([
            pd.DataFrame({
                "年度": [2024], "预测机构数": [30], "最小值": [50.0],
                "均值": [55.0], "最大值": [60.0], "行业平均数": [7.0],
            }),
            THS_TABLE,
        ], ignore_index=True)
        evidence = self.build(frame=stale)
        for period in evidence.periods.values():
            self.assertNotAlmostEqual(period.eps.current.value, 55.0)
        self.assertNotIn("-1y", evidence.periods)


class HonestGapTests(AShareTestCase):
    def setUp(self):
        super().setUp()
        self.evidence = self.build()

    def test_momentum_is_insufficient_on_a_first_run_not_neutral(self):
        self.assertEqual(self.evidence.momentum.band, "Insufficient Data")
        self.assertIsNone(self.evidence.momentum.score)

    def test_no_revision_history_at_any_horizon(self):
        eps = self.evidence.periods["0y"].eps
        for horizon in (eps.days_ago_7, eps.days_ago_30, eps.days_ago_60, eps.days_ago_90):
            self.assertFalse(horizon.available)
            self.assertIn("no revision history", horizon.unavailable_reason)

    def test_no_breadth_at_any_window(self):
        breadth = self.evidence.periods["0y"].breadth
        for window in ("7d", "30d", "90d"):
            self.assertIsNone(breadth.net_ratio(window))

    def test_no_revenue_estimate(self):
        revenue = self.evidence.periods["0y"].revenue
        self.assertFalse(revenue.current.available)
        self.assertIn("EPS only", revenue.current.unavailable_reason)

    def test_no_calendar_and_the_reason_names_the_filing_we_do_not_fetch(self):
        self.assertFalse(self.evidence.calendar.available)
        self.assertIn("预约披露时间", self.evidence.calendar.unavailable_reason)

    def test_no_surprises_and_no_drift(self):
        self.assertEqual(self.evidence.surprises, [])
        self.assertEqual(self.evidence.drift, [])
        self.assertIn("announcement dates", self.evidence.drift_unavailable_reason)

    def test_the_verbatim_table_travels_as_a_sourced_note(self):
        note = self.evidence.guidance[0]
        self.assertIn("年度", note.text)
        self.assertIn("67.85", note.text)
        self.assertIn("10jqka", note.url)
        self.assertIn("行业平均数", note.text)

    def test_the_verbatim_table_does_not_need_tabulate(self):
        """``DataFrame.to_markdown`` requires an undeclared dependency.

        It raises ImportError on a clean install, which was verified on this
        machine, so the note is assembled by hand instead.
        """
        import builtins

        real_import = builtins.__import__

        def no_tabulate(name, *args, **kwargs):
            if name == "tabulate":
                raise ImportError("No module named 'tabulate'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", no_tabulate):
            evidence = self.build()
        self.assertIn("67.85", evidence.guidance[0].text)

    def test_whisper_and_margin_gaps_are_declared(self):
        joined = " ".join(self.evidence.data_gaps)
        self.assertIn("Whisper expectations", joined)
        self.assertIn("margin revisions", joined)


class UnrecognisedLayoutTests(AShareTestCase):
    def test_an_unknown_layout_claims_no_numbers_and_says_why(self):
        mystery = pd.DataFrame({
            "预测期间": ["2026", "2027"],
            "某个数字": [10.0, 11.0],
            "另一个": [1.0, 2.0],
        })
        evidence = self.build(frame=mystery)
        self.assertEqual(evidence.periods, {})
        self.assertEqual(evidence.momentum.band, "Insufficient Data")
        joined = " ".join(evidence.data_gaps)
        self.assertIn("did not carry the expected 年度 / 均值 columns", joined)
        self.assertIn("realised result as a forward consensus", joined)
        # The numbers are still visible to a reader, just not claimed as consensus.
        self.assertIn("某个数字", evidence.guidance[0].text)

    def test_a_table_with_headers_but_no_usable_row_is_reported(self):
        empty_rows = pd.DataFrame({
            "年度": ["-", None], "预测机构数": [None, None],
            "均值": [None, "-"],
        })
        evidence = self.build(frame=empty_rows)
        self.assertEqual(evidence.periods, {})
        self.assertTrue(
            any("no row resolved" in g for g in evidence.data_gaps), evidence.data_gaps
        )

    def test_header_aliases_are_accepted(self):
        aliased = pd.DataFrame({
            "年份": [2026, 2027], "机构数": [40, 38],
            "平均值": [67.85, 71.79],
        })
        evidence = self.build(frame=aliased)
        self.assertAlmostEqual(evidence.periods["0y"].eps.current.value, 67.85)
        self.assertAlmostEqual(evidence.periods["0y"].analyst_count.value, 40)


class HistoricalDateTests(AShareTestCase):
    def test_a_past_trade_date_carries_a_future_function_warning(self):
        """This is today's snapshot, not the vintage that existed then."""
        evidence = self.build(as_of="2020-01-15")
        joined = " ".join(evidence.warnings)
        self.assertIn("未来函数警告", joined)
        self.assertIn("2020-01-15", joined)

    def test_todays_date_carries_no_such_warning(self):
        import datetime as dt

        today = dt.datetime.now(dt.timezone.utc).date().isoformat()
        evidence = self.build(as_of=today)
        self.assertFalse(any("未来函数警告" in w for w in evidence.warnings))


class LocalVintageTests(AShareTestCase):
    def test_a_run_records_a_snapshot_so_history_can_accrue(self):
        from tradingagents.dataflows.earnings_snapshot_store import default_store

        self.build()
        self.assertTrue(default_store().observed_dates("600519"))

    def test_an_unwritable_store_does_not_fail_the_run(self):
        from tradingagents.dataflows import earnings_snapshot_store as store_module

        with patch.object(
            store_module.EarningsSnapshotStore,
            "append",
            side_effect=store_module.SnapshotStoreError("disk full"),
        ):
            evidence = self.build()
        self.assertAlmostEqual(evidence.periods["0y"].eps.current.value, 67.85)


class JsonToolOutputTests(AShareTestCase):
    def test_the_routed_entry_point_returns_parseable_json(self):
        with patch.object(ase, "_eps_forecast_ths_frame", return_value=THS_TABLE):
            payload = json.loads(ase.get_earnings_evidence("600519", AS_OF))
        self.assertEqual(payload["symbol"], "600519")
        self.assertEqual(payload["currency"], "CNY")
        self.assertAlmostEqual(payload["periods"]["0y"]["eps"]["current"]["value"], 67.85)


if __name__ == "__main__":
    unittest.main()

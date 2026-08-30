"""yfinance earnings adapter: table normalization, routing refusal, drift math.

Fixture-driven. Every yfinance table is a fake, so the tests cover the shapes
that were observed live — including the ones that are internally inconsistent —
without a network call.
"""

import unittest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

from tradingagents.dataflows import yfinance_earnings as ye
from tradingagents.dataflows.errors import NoMarketDataError

# --------------------------------------------------------------------------
# Fixtures: verbatim shapes captured from yfinance 1.6.0.
# --------------------------------------------------------------------------

AAPL_EPS_TREND = pd.DataFrame(
    {
        "current": [1.97656, 2.90859, 8.81249, 9.53127],
        "7daysAgo": [1.97549, 2.90495, 8.80708, 9.53948],
        "30daysAgo": [2.01755, 2.96466, 8.76760, 9.71110],
        "60daysAgo": [2.00801, 2.94699, 8.75958, 9.67425],
        "90daysAgo": [2.00767, 2.94699, 8.75324, 9.65098],
        "currency": ["USD"] * 4,
    },
    index=pd.Index(["0q", "+1q", "0y", "+1y"], name="period"),
)

# Note the inconsistent capitalisation Yahoo actually ships: `upLast7days`
# (lowercase d) beside `downLast7Days` (uppercase D).
AAPL_EPS_REVISIONS = pd.DataFrame(
    {
        "upLast7days": [5, 2, 20, 6],
        "upLast30days": [7, 4, 21, 8],
        "downLast30days": [14, 11, 8, 19],
        "downLast7Days": [16, 14, 9, 21],
        "currency": ["USD"] * 4,
    },
    index=pd.Index(["0q", "+1q", "0y", "+1y"], name="period"),
)

AAPL_EARNINGS_ESTIMATE = pd.DataFrame(
    {
        "avg": [1.97656, 2.90859, 8.81249, 9.53127],
        "low": [1.93, 2.51, 8.28, 8.24],
        "high": [2.07, 3.42, 8.94, 10.67],
        "yearAgoEps": [1.85, 2.84, 7.46, 8.81249],
        "numberOfAnalysts": [28, 22, 37, 39],
        "growth": [0.0684, 0.0242, 0.1813, 0.0816],
        "currency": ["USD"] * 4,
    },
    index=pd.Index(["0q", "+1q", "0y", "+1y"], name="period"),
)

AAPL_REVENUE_ESTIMATE = pd.DataFrame(
    {
        "avg": [113550860190, 154432499960, 477683718840, 525003468150],
        "low": [112137000000, 132850061129, 471800000000, 483496000000],
        "high": [117219700000, 170414000000, 483750000000, 594863000000],
        "numberOfAnalysts": [28, 20, 36, 39],
        "yearAgoRevenue": [102466000000, 143756000000, 416161000000, 477683718840],
        "growth": [0.1082, 0.0743, 0.1478, 0.0991],
        "currency": ["USD"] * 4,
    },
    index=pd.Index(["0q", "+1q", "0y", "+1y"], name="period"),
)

AAPL_EARNINGS_HISTORY = pd.DataFrame(
    {
        "epsActual": [1.85, 2.84, 2.01, 2.02],
        "epsEstimate": [1.76993, 2.67080, 1.94275, 1.89243],
        "epsDifference": [0.08, 0.17, 0.07, 0.13],
        "surprisePercent": [0.0452, 0.0634, 0.0346, 0.0674],
    },
    index=pd.Index(
        ["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"], name="quarter"
    ),
)

AAPL_CALENDAR = {
    "Dividend Date": date(2026, 8, 12),
    "Ex-Dividend Date": date(2026, 8, 9),
    "Earnings Date": [date(2026, 10, 29)],
    "Earnings High": 2.07,
    "Earnings Low": 1.93,
    "Earnings Average": 1.98013,
    "Revenue High": 117219700000,
    "Revenue Low": 112137000000,
    "Revenue Average": 113550860190,
}

# 600519.SS: Yahoo returns an empty Earnings Date list and None estimates.
ASHARE_CALENDAR = {
    "Ex-Dividend Date": date(2026, 6, 25),
    "Earnings Date": [],
    "Earnings High": None,
    "Earnings Low": None,
    "Earnings Average": None,
    "Revenue High": 42000000000,
    "Revenue Low": 40606330000,
    "Revenue Average": 41303165000,
}

AAPL_INFO = {
    "quoteType": "EQUITY",
    "longName": "Apple Inc.",
    "shortName": "Apple Inc.",
    "currency": "USD",
    "financialCurrency": "USD",
    # 2026-09-27, as epoch seconds.
    "nextFiscalYearEnd": int(datetime(2026, 9, 27, tzinfo=timezone.utc).timestamp()),
}

EMPTY = pd.DataFrame()


class FakeTicker:
    """Stands in for ``yf.Ticker`` with a fixed set of tables."""

    def __init__(self, symbol, *, info=None, tables=None, calendar=None, earnings_dates=None):
        self.ticker = symbol
        self._info = info if info is not None else {}
        self._tables = tables or {}
        self._calendar = calendar if calendar is not None else {}
        self._earnings_dates = earnings_dates

    @property
    def info(self):
        return self._info

    @property
    def calendar(self):
        return self._calendar

    def __getattr__(self, name):
        tables = self.__dict__.get("_tables", {})
        if name in tables:
            value = tables[name]
            if isinstance(value, Exception):
                raise value
            return value
        raise AttributeError(name)

    def get_earnings_dates(self, limit=12):
        if isinstance(self._earnings_dates, Exception):
            raise self._earnings_dates
        if self._earnings_dates is None:
            raise RuntimeError("scrape unavailable")
        return self._earnings_dates


def _full_tables(**overrides):
    tables = {
        "earnings_estimate": AAPL_EARNINGS_ESTIMATE,
        "revenue_estimate": AAPL_REVENUE_ESTIMATE,
        "eps_trend": AAPL_EPS_TREND,
        "eps_revisions": AAPL_EPS_REVISIONS,
        "earnings_history": AAPL_EARNINGS_HISTORY,
    }
    tables.update(overrides)
    return tables


class AdapterTestCase(unittest.TestCase):
    """Isolates the snapshot store and the clock for every adapter test."""

    def setUp(self):
        from tempfile import TemporaryDirectory

        self._tmp = TemporaryDirectory()
        self._config = patch.dict(
            "tradingagents.dataflows.config._config",
            {"data_cache_dir": self._tmp.name},
        )
        self._config.start()
        # Today, so the live path is taken rather than the point-in-time path.
        self.today = datetime.now(timezone.utc).date().isoformat()

    def tearDown(self):
        self._config.stop()
        self._tmp.cleanup()

    def build(self, symbol="AAPL", *, as_of=None, ticker=None, **kwargs):
        fake = ticker or FakeTicker(symbol, **kwargs)
        with patch.object(ye.yf, "Ticker", return_value=fake):
            return ye.build_earnings_evidence(symbol, as_of or self.today)


class SymbolRefusalTests(AdapterTestCase):
    """Routing by refusal: this vendor must not answer emptily."""

    def test_bare_numeric_symbol_is_refused_without_a_network_call(self):
        def explode(*_args, **_kwargs):
            raise AssertionError("must not construct a Ticker for a bare numeric code")

        with patch.object(ye.yf, "Ticker", explode):
            for symbol in ("600519", "000660", "7203"):
                with self.subTest(symbol=symbol):
                    with self.assertRaises(NoMarketDataError) as ctx:
                        ye.build_earnings_evidence(symbol, self.today)
                    self.assertIn("venue suffix", str(ctx.exception))

    def test_unknown_symbol_raises_so_the_next_vendor_is_tried(self):
        with self.assertRaises(NoMarketDataError):
            self.build("ZZZQQQNOPE", info={}, tables=_full_tables(
                earnings_estimate=EMPTY, revenue_estimate=EMPTY, eps_trend=EMPTY,
                eps_revisions=EMPTY, earnings_history=EMPTY,
            ))

    def test_non_operating_symbol_forms_are_answered_not_refused(self):
        """No later vendor would do better, so this is a real answer."""
        def explode(*_args, **_kwargs):
            raise AssertionError("symbol form should settle it with no network call")

        cases = [("^GSPC", "index"), ("EURUSD=X", "exchange"), ("GC=F", "futures"),
                 ("BTC-USD", "crypto")]
        with patch.object(ye.yf, "Ticker", explode):
            for symbol, needle in cases:
                with self.subTest(symbol=symbol):
                    evidence = ye.build_earnings_evidence(symbol, self.today)
                    self.assertEqual(evidence.status, "unsupported")
                    self.assertIn(needle, evidence.status_detail.lower())

    def test_etf_quote_type_is_unsupported_rather_than_no_data(self):
        evidence = self.build(
            "SPY",
            info={"quoteType": "ETF", "shortName": "SPDR S&P 500"},
            tables=_full_tables(),
            calendar={},
        )
        self.assertEqual(evidence.status, "unsupported")
        self.assertIn("etf", evidence.status_detail.lower())
        self.assertEqual(evidence.periods, {})

    def test_known_company_with_no_estimates_is_no_coverage(self):
        evidence = self.build(
            "TINYCO",
            info={"quoteType": "EQUITY", "shortName": "Tiny Co"},
            tables=_full_tables(
                earnings_estimate=EMPTY, revenue_estimate=EMPTY, eps_trend=EMPTY,
                eps_revisions=EMPTY, earnings_history=EMPTY,
            ),
            calendar={},
        )
        self.assertEqual(evidence.status, "no_coverage")
        self.assertIn("absence of sell-side", evidence.status_detail)
        self.assertIn("not a zero", evidence.status_detail)


class NormalizationTests(AdapterTestCase):
    def setUp(self):
        super().setUp()
        self.evidence = self.build(
            info=AAPL_INFO, tables=_full_tables(), calendar=AAPL_CALENDAR
        )

    def test_all_four_periods_are_present(self):
        self.assertEqual(sorted(self.evidence.periods), sorted(["0q", "+1q", "0y", "+1y"]))

    def test_eps_trend_horizons_are_read_from_the_trend_table(self):
        eps = self.evidence.periods["0y"].eps
        self.assertAlmostEqual(eps.current.value, 8.81249)
        self.assertAlmostEqual(eps.days_ago_7.value, 8.80708)
        self.assertAlmostEqual(eps.days_ago_30.value, 8.76760)
        self.assertAlmostEqual(eps.days_ago_60.value, 8.75958)
        self.assertAlmostEqual(eps.days_ago_90.value, 8.75324)

    def test_the_inconsistent_down_seven_day_column_is_still_found(self):
        """`downLast7Days` has a capital D where `upLast7days` does not.

        A case-sensitive lookup silently loses this column and reports one-sided
        breadth, which reads as unanimous upgrades.
        """
        breadth = self.evidence.periods["0y"].breadth
        self.assertAlmostEqual(breadth.up_7d.value, 20)
        self.assertAlmostEqual(breadth.down_7d.value, 9)
        self.assertAlmostEqual(breadth.up_30d.value, 21)
        self.assertAlmostEqual(breadth.down_30d.value, 8)

    def test_ninety_day_breadth_is_unavailable_not_copied_from_thirty_day(self):
        breadth = self.evidence.periods["0y"].breadth
        self.assertFalse(breadth.up_90d.available)
        self.assertFalse(breadth.down_90d.available)
        self.assertIn("no 90-day revision counts", breadth.up_90d.unavailable_reason)

    def test_revenue_history_is_unavailable_not_borrowed_from_eps(self):
        revenue = self.evidence.periods["0y"].revenue
        self.assertTrue(revenue.current.available)
        for horizon in (revenue.days_ago_7, revenue.days_ago_30, revenue.days_ago_90):
            self.assertFalse(horizon.available)
            self.assertIn("no revenue revision history", horizon.unavailable_reason)

    def test_analyst_count_and_year_ago_eps(self):
        period = self.evidence.periods["0y"]
        self.assertAlmostEqual(period.analyst_count.value, 37)
        self.assertAlmostEqual(period.year_ago_eps.value, 7.46)

    def test_fiscal_year_end_is_resolved_from_provider_metadata(self):
        self.assertEqual(self.evidence.periods["0y"].period.end_date, "2026-09-27")
        self.assertEqual(self.evidence.periods["+1y"].period.end_date, "2027-09-27")
        self.assertIn("FY2026", self.evidence.periods["0y"].period.label)

    def test_missing_fiscal_metadata_keeps_the_relative_label(self):
        evidence = self.build(
            info={"quoteType": "EQUITY", "shortName": "X"},
            tables=_full_tables(), calendar=AAPL_CALENDAR,
        )
        self.assertIsNone(evidence.periods["0y"].period.end_date)
        self.assertIn("relative period 0y", evidence.periods["0y"].period.label)
        self.assertTrue(any("No FY number is invented" in g for g in evidence.data_gaps))

    def test_yahoos_growth_column_is_not_used_as_a_change(self):
        """It is sign-broken on negative EPS, so nothing may read it."""
        momentum = self.evidence.momentum
        for value in momentum.signals.values():
            self.assertNotAlmostEqual(value, 0.1813, places=4)

    def test_currency_is_carried_onto_values(self):
        self.assertEqual(self.evidence.currency, "USD")
        self.assertEqual(self.evidence.periods["0y"].eps.current.currency, "USD")

    def test_momentum_matches_the_pure_computation(self):
        from tradingagents.dataflows.earnings_models import compute_momentum

        expected = compute_momentum(self.evidence.periods["0y"])
        self.assertEqual(self.evidence.momentum.band, expected.band)
        self.assertAlmostEqual(self.evidence.momentum.score, expected.score)

    def test_unavailable_by_design_gaps_are_always_declared(self):
        joined = " ".join(self.evidence.data_gaps)
        self.assertIn("Whisper expectations", joined)
        self.assertIn("margin revisions", joined)


class DegradedTableTests(AdapterTestCase):
    def test_one_missing_table_does_not_sink_the_request(self):
        evidence = self.build(
            info=AAPL_INFO,
            tables=_full_tables(revenue_estimate=EMPTY),
            calendar=AAPL_CALENDAR,
        )
        self.assertIn(evidence.status, {"ok", "partial"})
        self.assertTrue(evidence.periods["0y"].eps.current.available)
        self.assertFalse(evidence.periods["0y"].revenue.current.available)

    def test_a_raising_table_is_treated_as_absent(self):
        evidence = self.build(
            info=AAPL_INFO,
            tables=_full_tables(eps_revisions=RuntimeError("yahoo 500")),
            calendar=AAPL_CALENDAR,
        )
        self.assertFalse(evidence.periods["0y"].breadth.up_30d.available)
        self.assertTrue(evidence.periods["0y"].eps.current.available)

    def test_nan_cells_become_unavailable_not_zero(self):
        trend = AAPL_EPS_TREND.copy()
        trend.loc["0y", "30daysAgo"] = float("nan")
        evidence = self.build(
            info=AAPL_INFO, tables=_full_tables(eps_trend=trend), calendar=AAPL_CALENDAR
        )
        cell = evidence.periods["0y"].eps.days_ago_30
        self.assertFalse(cell.available)
        self.assertIsNone(cell.value)

    def test_current_eps_falls_back_to_the_estimate_table(self):
        trend = AAPL_EPS_TREND.drop(columns=["current"])
        evidence = self.build(
            info=AAPL_INFO, tables=_full_tables(eps_trend=trend), calendar=AAPL_CALENDAR
        )
        self.assertAlmostEqual(evidence.periods["0y"].eps.current.value, 8.81249)

    def test_a_long_term_growth_row_is_not_turned_into_a_period(self):
        trend = pd.concat([
            AAPL_EPS_TREND,
            pd.DataFrame(
                {"current": [float("nan")], "currency": ["USD"]},
                index=pd.Index(["LTG"], name="period"),
            ),
        ])
        evidence = self.build(
            info=AAPL_INFO, tables=_full_tables(eps_trend=trend), calendar=AAPL_CALENDAR
        )
        self.assertNotIn("LTG", evidence.periods)


class CalendarTests(AdapterTestCase):
    def test_confirmed_single_date(self):
        evidence = self.build(info=AAPL_INFO, tables=_full_tables(), calendar=AAPL_CALENDAR)
        self.assertEqual(evidence.calendar.next_date, "2026-10-29")
        self.assertFalse(evidence.calendar.date_is_estimated)
        self.assertAlmostEqual(evidence.calendar.eps_estimate_avg.value, 1.98013)
        self.assertAlmostEqual(evidence.calendar.revenue_estimate_avg.value, 113550860190)

    def test_two_dates_are_preserved_as_an_unconfirmed_window(self):
        calendar = dict(AAPL_CALENDAR)
        calendar["Earnings Date"] = [date(2026, 11, 3), date(2026, 11, 7)]
        evidence = self.build(info=AAPL_INFO, tables=_full_tables(), calendar=calendar)
        self.assertEqual(evidence.calendar.next_date, "2026-11-03")
        self.assertEqual(evidence.calendar.next_date_range_end, "2026-11-07")
        self.assertTrue(evidence.calendar.date_is_estimated)

    def test_an_empty_date_list_is_unavailable_with_a_reason(self):
        evidence = self.build(
            "600519.SS", info=AAPL_INFO, tables=_full_tables(), calendar=ASHARE_CALENDAR
        )
        self.assertFalse(evidence.calendar.available)
        self.assertIn("no upcoming earnings date", evidence.calendar.unavailable_reason)

    def test_release_timing_is_unknown_rather_than_assumed(self):
        """Guessing inverts the drift anchor by a day."""
        evidence = self.build(info=AAPL_INFO, tables=_full_tables(), calendar=AAPL_CALENDAR)
        self.assertEqual(evidence.calendar.timing, "unknown")

    def test_none_estimates_are_unavailable_not_zero(self):
        evidence = self.build(
            "600519.SS", info=AAPL_INFO, tables=_full_tables(), calendar=ASHARE_CALENDAR
        )
        self.assertFalse(evidence.calendar.eps_estimate_avg.available)
        self.assertIsNone(evidence.calendar.eps_estimate_avg.value)


class SurpriseTests(AdapterTestCase):
    def test_history_is_read_with_quarter_ends_and_no_announcement_dates(self):
        evidence = self.build(info=AAPL_INFO, tables=_full_tables(), calendar=AAPL_CALENDAR)
        self.assertEqual(len(evidence.surprises), 4)
        first = evidence.surprises[0]
        self.assertEqual(first.fiscal_period_end, "2025-09-30")
        self.assertAlmostEqual(first.eps_actual.value, 1.85)
        self.assertAlmostEqual(first.surprise_pct.value, 0.0452)

    def test_announcement_dates_are_absent_when_the_scrape_fails(self):
        evidence = self.build(info=AAPL_INFO, tables=_full_tables(), calendar=AAPL_CALENDAR)
        self.assertTrue(all(s.announcement_date is None for s in evidence.surprises))

    def test_surprises_are_sorted_oldest_first(self):
        evidence = self.build(info=AAPL_INFO, tables=_full_tables(), calendar=AAPL_CALENDAR)
        ends = [s.fiscal_period_end for s in evidence.surprises]
        self.assertEqual(ends, sorted(ends))

    def test_beat_flag_follows_the_difference(self):
        evidence = self.build(info=AAPL_INFO, tables=_full_tables(), calendar=AAPL_CALENDAR)
        self.assertTrue(all(s.beat for s in evidence.surprises))


class DriftUnavailabilityTests(AdapterTestCase):
    def test_drift_is_not_anchored_to_a_fiscal_quarter_end(self):
        """A June quarter is announced in late July.

        Anchoring there would report weeks of unrelated trading as the earnings
        reaction, so with no announcement dates drift must be absent.
        """
        evidence = self.build(info=AAPL_INFO, tables=_full_tables(), calendar=AAPL_CALENDAR)
        self.assertEqual(evidence.drift, [])
        reason = evidence.drift_unavailable_reason
        self.assertIn("announced in late July", reason)
        self.assertIn("HTML scrape", reason)


class AnnouncementMatchingTests(unittest.TestCase):
    def test_a_quarter_is_matched_to_the_first_announcement_after_it(self):
        """Matching to the nearest date in either direction picks the wrong one.

        The previous quarter's release is usually closer to a given quarter end
        than that quarter's own release.
        """
        from tradingagents.dataflows.earnings_models import SurpriseEvent

        events = [SurpriseEvent(fiscal_period_end="2026-06-30")]
        announcements = ["2026-05-01", "2026-07-31", "2026-10-29"]
        matched = ye._match_announcements(events, announcements)
        self.assertEqual(matched[0].announcement_date, "2026-07-31")

    def test_a_quarter_with_no_later_announcement_stays_unmatched(self):
        from tradingagents.dataflows.earnings_models import SurpriseEvent

        matched = ye._match_announcements(
            [SurpriseEvent(fiscal_period_end="2026-06-30")], ["2026-05-01"]
        )
        self.assertIsNone(matched[0].announcement_date)


class DriftWindowTests(unittest.TestCase):
    """Pure arithmetic, so every calendar edge is testable from a fixture."""

    def setUp(self):
        self.days = pd.bdate_range("2026-01-05", periods=40)
        self.prices = pd.DataFrame(
            {"Date": self.days, "Close": [100 * (1.01**i) for i in range(40)]}
        )
        self.bench = pd.DataFrame(
            {"Date": self.days, "Close": [100 * (1.002**i) for i in range(40)]}
        )

    def test_anchor_is_the_first_session_after_the_announcement(self):
        # Friday 2026-01-09 announcement -> Monday 2026-01-12 anchor.
        obs = ye.compute_drift_windows(
            self.prices, [("2025-12-31", "2026-01-09")], horizons=(1,)
        )
        self.assertEqual(obs[0].anchor_session, "2026-01-12")

    def test_a_weekend_announcement_anchors_to_the_next_session(self):
        for announced in ("2026-01-10", "2026-01-11"):
            with self.subTest(announced=announced):
                obs = ye.compute_drift_windows(
                    self.prices, [("2025-12-31", announced)], horizons=(1,)
                )
                self.assertEqual(obs[0].anchor_session, "2026-01-12")

    def test_a_market_holiday_gap_anchors_to_the_next_traded_session(self):
        gapped = self.prices[self.prices["Date"] != pd.Timestamp("2026-01-12")].reset_index(
            drop=True
        )
        obs = ye.compute_drift_windows(
            gapped, [("2025-12-31", "2026-01-09")], horizons=(1,)
        )
        self.assertEqual(obs[0].anchor_session, "2026-01-13")

    def test_horizons_are_measured_in_sessions(self):
        obs = ye.compute_drift_windows(
            self.prices, [("2025-12-31", "2026-01-09")], horizons=(1, 5, 20)
        )
        got = {o.sessions: round(o.stock_return.value, 6) for o in obs}
        self.assertAlmostEqual(got[1], round(1.01**1 - 1, 6))
        self.assertAlmostEqual(got[5], round(1.01**5 - 1, 6))
        self.assertAlmostEqual(got[20], round(1.01**20 - 1, 6))

    def test_a_window_running_past_the_data_is_omitted_not_truncated(self):
        """Reporting a 12-session move as the 20-session drift would be worse."""
        obs = ye.compute_drift_windows(
            self.prices, [("2025-12-31", "2026-01-09")], horizons=(1, 5, 20, 60)
        )
        self.assertEqual(sorted(o.sessions for o in obs), [1, 5, 20])

    def test_benchmark_is_aligned_by_date_not_by_session_count(self):
        """A benchmark holiday must not shift its window.

        With session-count alignment a benchmark missing one session reported
        1.206% where the correct figure over the stock's calendar window is
        1.004%.
        """
        aligned = ye.compute_drift_windows(
            self.prices, [("2025-12-31", "2026-01-09")], benchmark=self.bench, horizons=(5,)
        )
        gapped_bench = self.bench[
            self.bench["Date"] != pd.Timestamp("2026-01-13")
        ].reset_index(drop=True)
        gapped = ye.compute_drift_windows(
            self.prices, [("2025-12-31", "2026-01-09")], benchmark=gapped_bench, horizons=(5,)
        )
        self.assertAlmostEqual(aligned[0].benchmark_return.value, 1.002**5 - 1, places=6)
        self.assertAlmostEqual(
            gapped[0].benchmark_return.value, aligned[0].benchmark_return.value, places=6
        )

    def test_excess_return_is_the_difference(self):
        obs = ye.compute_drift_windows(
            self.prices, [("2025-12-31", "2026-01-09")], benchmark=self.bench, horizons=(5,)
        )
        self.assertAlmostEqual(
            obs[0].excess_return.value,
            obs[0].stock_return.value - obs[0].benchmark_return.value,
        )

    def test_no_benchmark_leaves_excess_unavailable_rather_than_zero(self):
        obs = ye.compute_drift_windows(
            self.prices, [("2025-12-31", "2026-01-09")], horizons=(5,)
        )
        self.assertFalse(obs[0].benchmark_return.available)
        self.assertFalse(obs[0].excess_return.available)
        self.assertIsNone(obs[0].excess_return.value)

    def test_an_announcement_after_all_data_yields_nothing(self):
        self.assertEqual(
            ye.compute_drift_windows(
                self.prices, [("2026-01-01", "2026-12-01")], horizons=(1,)
            ),
            [],
        )

    def test_events_are_capped_to_the_most_recent(self):
        events = [
            ("2025-03-31", "2026-01-06"),
            ("2025-06-30", "2026-01-07"),
            ("2025-09-30", "2026-01-08"),
            ("2025-12-31", "2026-01-09"),
            ("2026-03-31", "2026-01-12"),
        ]
        obs = ye.compute_drift_windows(self.prices, events, horizons=(1,), max_events=2)
        self.assertEqual(
            sorted({o.announcement_date for o in obs}), ["2026-01-09", "2026-01-12"]
        )

    def test_empty_or_unusable_price_frames_yield_nothing(self):
        self.assertEqual(ye.compute_drift_windows(pd.DataFrame(), [("a", "b")]), [])
        nan_only = pd.DataFrame({"Date": self.days[:3], "Close": [None, None, None]})
        self.assertEqual(
            ye.compute_drift_windows(nan_only, [("2025-12-31", "2026-01-05")]), []
        )

    def test_an_unparseable_announcement_date_is_skipped(self):
        self.assertEqual(
            ye.compute_drift_windows(
                self.prices, [("2025-12-31", "not-a-date")], horizons=(1,)
            ),
            [],
        )


class PointInTimeRoutingTests(AdapterTestCase):
    def test_a_historical_date_never_reaches_a_live_endpoint(self):
        def explode(*_args, **_kwargs):
            raise AssertionError("a historical date must not hit a live estimates call")

        with patch.object(ye.yf, "Ticker", explode):
            evidence = ye.build_earnings_evidence("AAPL", "2020-01-15")
        self.assertEqual(evidence.status, "pit_unavailable")
        self.assertIn("relative to the present", evidence.status_detail)

    def test_a_historical_date_with_a_vintage_serves_the_vintage(self):
        from tradingagents.dataflows.earnings_snapshot_store import default_store

        live = self.build(info=AAPL_INFO, tables=_full_tables(), calendar=AAPL_CALENDAR)
        past = (date.fromisoformat(self.today) - timedelta(days=40)).isoformat()
        default_store().append(live, observed_date=past)

        asked = (date.fromisoformat(self.today) - timedelta(days=10)).isoformat()

        def explode(*_args, **_kwargs):
            raise AssertionError("must be served from the store")

        with patch.object(ye.yf, "Ticker", explode):
            evidence = ye.build_earnings_evidence("AAPL", asked)

        self.assertNotEqual(evidence.status, "pit_unavailable")
        self.assertEqual(evidence.as_of, asked)
        self.assertAlmostEqual(evidence.periods["0y"].eps.current.value, 8.81249)
        self.assertTrue(any(f"observed {past}" in s for s in evidence.sources))

    def test_is_historical_is_generous_about_today(self):
        """A timezone offset must not divert a same-day run onto the PIT path."""
        self.assertFalse(ye._is_historical(self.today))
        tomorrow = (date.fromisoformat(self.today) + timedelta(days=1)).isoformat()
        self.assertFalse(ye._is_historical(tomorrow))
        self.assertTrue(ye._is_historical("2001-01-01"))
        self.assertFalse(ye._is_historical("garbage"))


class PersistenceTests(AdapterTestCase):
    def test_a_live_fetch_records_a_vintage_keyed_by_the_real_observation_date(self):
        """Never keyed by the requested date.

        Keying on the fetch date makes it structurally impossible for a timezone
        boundary to file today's consensus as an earlier vintage, which would
        corrupt every later historical read.
        """
        from tradingagents.dataflows.earnings_snapshot_store import default_store

        self.build(info=AAPL_INFO, tables=_full_tables(), calendar=AAPL_CALENDAR)
        self.assertEqual(default_store().observed_dates("AAPL"), [self.today])

    def test_an_unwritable_store_does_not_fail_a_run_that_has_its_answer(self):
        from tradingagents.dataflows import earnings_snapshot_store as store_module

        with patch.object(
            store_module.EarningsSnapshotStore,
            "append",
            side_effect=store_module.SnapshotStoreError("disk full"),
        ):
            evidence = self.build(
                info=AAPL_INFO, tables=_full_tables(), calendar=AAPL_CALENDAR
            )
        self.assertIn(evidence.status, {"ok", "partial"})
        self.assertTrue(evidence.periods["0y"].eps.current.available)


class JsonToolOutputTests(AdapterTestCase):
    def test_the_routed_entry_point_returns_parseable_json(self):
        import json

        fake = FakeTicker("AAPL", info=AAPL_INFO, tables=_full_tables(), calendar=AAPL_CALENDAR)
        with patch.object(ye.yf, "Ticker", return_value=fake):
            raw = ye.get_earnings_evidence("AAPL", self.today)
        payload = json.loads(raw)
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertIn("momentum", payload)
        self.assertIn("periods", payload)


if __name__ == "__main__":
    unittest.main()

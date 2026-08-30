"""Alpha Vantage earnings adapter: entitlements, date filtering, transcript selection.

Every request is faked. The point of this vendor is the two things Yahoo cannot
give — real ``reportedDate`` values and ``reportTime`` — so the tests concentrate
on the announcement-date filtering that depends on them, and on the degradation
paths a free-tier key actually hits.
"""

import json
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tradingagents.dataflows import alpha_vantage_earnings as av
from tradingagents.dataflows.alpha_vantage_common import (
    AlphaVantageNotConfiguredError,
    AlphaVantageRateLimitError,
)
from tradingagents.dataflows.errors import NoMarketDataError

EARNINGS_PAYLOAD = {
    "symbol": "IBM",
    "annualEarnings": [
        {"fiscalDateEnding": "2025-12-31", "reportedEPS": "9.99"},
    ],
    "quarterlyEarnings": [
        {"fiscalDateEnding": "2026-06-30", "reportedDate": "2026-07-22",
         "reportedEPS": "2.80", "estimatedEPS": "2.62", "surprise": "0.18",
         "surprisePercentage": "6.870", "reportTime": "post-market"},
        {"fiscalDateEnding": "2026-03-31", "reportedDate": "2026-04-23",
         "reportedEPS": "1.60", "estimatedEPS": "1.58", "surprise": "0.02",
         "surprisePercentage": "1.266", "reportTime": "post-market"},
        {"fiscalDateEnding": "2025-12-31", "reportedDate": "2026-01-28",
         "reportedEPS": "3.92", "estimatedEPS": "3.78", "surprise": "0.14",
         "surprisePercentage": "3.704", "reportTime": "post-market"},
        # Announced after the analysis date used below: must be filtered out.
        {"fiscalDateEnding": "2026-09-30", "reportedDate": "2026-10-22",
         "reportedEPS": "2.90", "estimatedEPS": "2.75", "surprise": "0.15",
         "surprisePercentage": "5.454", "reportTime": "post-market"},
    ],
}

ESTIMATES_PAYLOAD = {
    "symbol": "IBM",
    "estimates": [
        {"horizon": "current fiscal year", "date": "2026-12-31", "currency": "USD",
         "eps_estimate_average": "11.20", "eps_estimate_high": "11.60",
         "eps_estimate_low": "10.80", "eps_estimate_analyst_count": "17",
         "eps_estimate_average_7_days_ago": "11.15",
         "eps_estimate_average_30_days_ago": "11.02",
         "eps_estimate_average_60_days_ago": "10.95",
         "eps_estimate_average_90_days_ago": "10.90",
         "eps_estimate_revision_up_trailing_7_days": "3",
         "eps_estimate_revision_down_trailing_7_days": "1",
         "eps_estimate_revision_up_trailing_30_days": "9",
         "eps_estimate_revision_down_trailing_30_days": "2",
         "revenue_estimate_average": "64500000000",
         "revenue_estimate_average_30_days_ago": "64100000000"},
        {"horizon": "next fiscal year", "date": "2027-12-31", "currency": "USD",
         "eps_estimate_average": "12.40", "eps_estimate_analyst_count": "15",
         "eps_estimate_average_30_days_ago": "12.10"},
        # An unrecognised horizon must be skipped, not assigned a period key.
        {"horizon": "next fiscal year+2", "date": "2029-12-31",
         "eps_estimate_average": "15.00"},
    ],
}

CALENDAR_CSV = (
    "symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"
    "IBM,International Business Machines,2026-10-22,2026-09-30,2.75,USD\n"
    "IBM,International Business Machines,2027-01-27,2026-12-31,4.10,USD\n"
    "MSFT,Microsoft Corporation,2026-10-28,2026-09-30,3.55,USD\n"
)

TRANSCRIPT_PAYLOAD = {
    "symbol": "IBM",
    "quarter": "2026Q2",
    "transcript": [
        {"speaker": "Arvind Krishna", "title": "CEO",
         "content": "We are raising our full-year free cash flow guidance.",
         "sentiment": "0.7"},
        {"speaker": "Jim Kavanaugh", "title": "CFO",
         "content": "Software revenue grew 9% at constant currency.",
         "sentiment": "0.6"},
        {"speaker": "Operator", "title": "", "content": ""},
    ],
}

AS_OF = "2026-08-30"


def _responder(mapping):
    """Fake ``_make_api_request``: dict/str per function name, or raise."""
    def call(function_name, params):
        value = mapping.get(function_name)
        if value is None:
            raise AssertionError(f"unexpected Alpha Vantage call: {function_name}")
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, str) else json.dumps(value)
    return call


class AlphaVantageTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._config = patch.dict(
            "tradingagents.dataflows.config._config", {"data_cache_dir": self._tmp.name}
        )
        self._config.start()
        # Drift needs a price history; suppress it so these tests stay offline.
        self._prices = patch.object(
            av, "_attach_drift", side_effect=lambda evidence, *_a, **_k: evidence
        )
        self._prices.start()

    def tearDown(self):
        self._prices.stop()
        self._config.stop()
        self._tmp.cleanup()

    def build(self, mapping, symbol="IBM", as_of=AS_OF):
        with patch.object(av, "_make_api_request", _responder(mapping)):
            return av.build_earnings_evidence(symbol, as_of)


class ConfigurationTests(unittest.TestCase):
    def test_no_key_raises_before_any_network_call(self):
        def explode(*_a, **_k):
            raise AssertionError("must not call the API without a key")

        with patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": ""}, clear=False), \
             patch.object(av, "_make_api_request", explode):
            with self.assertRaises(AlphaVantageNotConfiguredError):
                av.build_earnings_evidence("IBM", AS_OF)
            with self.assertRaises(AlphaVantageNotConfiguredError):
                av.get_earnings_commentary("IBM", AS_OF)


class SurpriseFilteringTests(AlphaVantageTestCase):
    def setUp(self):
        super().setUp()
        self.evidence = self.build({
            "EARNINGS": EARNINGS_PAYLOAD,
            "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })

    def test_filtering_is_on_the_announcement_date_not_the_fiscal_end(self):
        """On 2026-08-30 the June quarter is announced; the September one is not.

        A fiscal-end filter would admit a quarter that had ended but had not been
        reported, leaking a result that was not yet public.
        """
        ends = [s.fiscal_period_end for s in self.evidence.surprises]
        self.assertIn("2026-06-30", ends)
        self.assertNotIn("2026-09-30", ends)

    def test_announcement_dates_are_carried(self):
        by_end = {s.fiscal_period_end: s for s in self.evidence.surprises}
        self.assertEqual(by_end["2026-06-30"].announcement_date, "2026-07-22")

    def test_surprise_percentage_is_normalized_to_a_decimal_fraction(self):
        """Alpha Vantage publishes '6.870'; yfinance publishes 0.0687.

        Without the conversion the same beat renders as 687%.
        """
        by_end = {s.fiscal_period_end: s for s in self.evidence.surprises}
        self.assertAlmostEqual(by_end["2026-06-30"].surprise_pct.value, 0.06870, places=5)
        self.assertEqual(by_end["2026-06-30"].surprise_pct.unit, "pct_dec")

    def test_reported_and_estimated_eps_are_read(self):
        by_end = {s.fiscal_period_end: s for s in self.evidence.surprises}
        self.assertAlmostEqual(by_end["2026-06-30"].eps_actual.value, 2.80)
        self.assertAlmostEqual(by_end["2026-06-30"].eps_estimate.value, 2.62)
        self.assertAlmostEqual(by_end["2026-06-30"].eps_difference.value, 0.18)

    def test_a_quarter_without_a_reported_date_falls_back_to_the_fiscal_end(self):
        payload = {
            "symbol": "X",
            "quarterlyEarnings": [
                {"fiscalDateEnding": "2026-06-30", "reportedEPS": "1.0"},
                {"fiscalDateEnding": "2026-12-31", "reportedEPS": "1.5"},
            ],
        }
        evidence = self.build({
            "EARNINGS": payload, "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })
        ends = [s.fiscal_period_end for s in evidence.surprises]
        self.assertEqual(ends, ["2026-06-30"])
        self.assertIsNone(evidence.surprises[0].announcement_date)

    def test_snake_case_keys_resolve_too(self):
        payload = {
            "symbol": "X",
            "quarterlyEarnings": [
                {"fiscal_date_ending": "2026-06-30", "reported_date": "2026-07-22",
                 "reported_eps": "2.80", "estimated_eps": "2.62",
                 "surprise_percentage": "6.870"},
            ],
        }
        evidence = self.build({
            "EARNINGS": payload, "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })
        self.assertEqual(evidence.surprises[0].announcement_date, "2026-07-22")
        self.assertAlmostEqual(evidence.surprises[0].eps_actual.value, 2.80)


class ReportTimingTests(AlphaVantageTestCase):
    def test_release_timing_is_resolved_from_the_most_recent_quarter(self):
        """This is what Yahoo leaves 'unknown'."""
        evidence = self.build({
            "EARNINGS": EARNINGS_PAYLOAD,
            "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })
        self.assertIn("amc", evidence.calendar.timing)
        self.assertIn("observed pattern", evidence.calendar.timing)

    def test_timing_stays_unknown_when_the_field_is_absent(self):
        payload = {
            "symbol": "X",
            "quarterlyEarnings": [
                {"fiscalDateEnding": "2026-06-30", "reportedDate": "2026-07-22",
                 "reportedEPS": "1.0"},
            ],
        }
        evidence = self.build({
            "EARNINGS": payload, "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })
        self.assertEqual(evidence.calendar.timing, "unknown")

    def test_pre_market_maps_to_bmo(self):
        payload = {
            "symbol": "X",
            "quarterlyEarnings": [
                {"fiscalDateEnding": "2026-06-30", "reportedDate": "2026-07-22",
                 "reportedEPS": "1.0", "reportTime": "pre-market"},
            ],
        }
        evidence = self.build({
            "EARNINGS": payload, "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })
        self.assertIn("bmo", evidence.calendar.timing)


class EstimateTests(AlphaVantageTestCase):
    def setUp(self):
        super().setUp()
        self.evidence = self.build({
            "EARNINGS": EARNINGS_PAYLOAD,
            "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })

    def test_horizons_map_onto_the_shared_period_keys(self):
        self.assertEqual(sorted(self.evidence.periods), sorted(["0y", "+1y"]))

    def test_unrecognised_horizons_are_skipped_not_guessed(self):
        """A new Alpha Vantage horizon must not overwrite the scored period."""
        for period in self.evidence.periods.values():
            self.assertNotAlmostEqual(period.eps.current.value or 0.0, 15.00)

    def test_eps_lookbacks_and_breadth_are_read(self):
        period = self.evidence.periods["0y"]
        self.assertAlmostEqual(period.eps.current.value, 11.20)
        self.assertAlmostEqual(period.eps.days_ago_7.value, 11.15)
        self.assertAlmostEqual(period.eps.days_ago_30.value, 11.02)
        self.assertAlmostEqual(period.eps.days_ago_90.value, 10.90)
        self.assertAlmostEqual(period.breadth.up_30d.value, 9)
        self.assertAlmostEqual(period.breadth.down_30d.value, 2)
        self.assertAlmostEqual(period.analyst_count.value, 17)

    def test_revenue_revision_history_is_available_from_this_vendor(self):
        period = self.evidence.periods["0y"]
        self.assertAlmostEqual(period.revenue.current.value, 64500000000)
        self.assertAlmostEqual(period.revenue.days_ago_30.value, 64100000000)

    def test_ninety_day_breadth_is_still_unavailable(self):
        breadth = self.evidence.periods["0y"].breadth
        self.assertFalse(breadth.up_90d.available)
        self.assertIn("7- and 30-day", breadth.up_90d.unavailable_reason)

    def test_momentum_is_scored_from_the_current_fiscal_year(self):
        self.assertEqual(self.evidence.momentum.period_key, "0y")
        self.assertEqual(self.evidence.momentum.band, "Positive")
        self.assertAlmostEqual(self.evidence.momentum.score, 0.391202, places=5)

    def test_this_vendor_can_reach_full_signal_coverage(self):
        """The only path to 1.00 weight: it publishes a revenue revision history.

        yfinance caps at 0.90 because it has no revenue trend at all, so the
        difference is a real capability gap rather than a fixture artefact.
        """
        self.assertAlmostEqual(self.evidence.momentum.available_weight, 1.0)
        self.assertEqual(
            sorted(self.evidence.momentum.signals),
            ["breadth_30d", "eps_30d", "eps_7d", "eps_90d", "revenue_30d"],
        )
        self.assertEqual(self.evidence.momentum.confidence, "high")

    def test_absent_horizons_on_a_sparser_row_are_unavailable_not_zero(self):
        period = self.evidence.periods["+1y"]
        self.assertTrue(period.eps.days_ago_30.available)
        self.assertFalse(period.eps.days_ago_7.available)
        self.assertIsNone(period.eps.days_ago_7.value)


class EntitlementDegradationTests(AlphaVantageTestCase):
    def test_a_premium_notice_on_estimates_keeps_the_surprise_history(self):
        """The free tier reports premium gating as a quota notice.

        Letting it propagate would discard the announcement dates and surprises
        already in hand over a field the user was never entitled to.
        """
        evidence = self.build({
            "EARNINGS": EARNINGS_PAYLOAD,
            "EARNINGS_ESTIMATES": AlphaVantageRateLimitError("premium endpoint"),
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })
        self.assertEqual(evidence.periods, {})
        self.assertEqual(len(evidence.surprises), 3)
        self.assertEqual(evidence.momentum.band, "Insufficient Data")
        joined = " ".join(evidence.data_gaps)
        self.assertIn("premium-gated", joined)
        self.assertIn("entitlement limit, not an absence of analyst coverage", joined)

    def test_a_failure_on_the_core_endpoint_propagates(self):
        """It is the core payload; swallowing it would report a false absence."""
        with patch.object(av, "_make_api_request", _responder({
            "EARNINGS": AlphaVantageRateLimitError("daily quota exhausted"),
        })):
            with self.assertRaises(AlphaVantageRateLimitError):
                av.build_earnings_evidence("IBM", AS_OF)

    def test_a_malformed_core_payload_raises_no_market_data(self):
        for payload in ("not json at all", {}, {"symbol": "IBM"}):
            with self.subTest(payload=payload):
                with patch.object(av, "_make_api_request", _responder({"EARNINGS": payload})):
                    with self.assertRaises(NoMarketDataError):
                        av.build_earnings_evidence("IBM", AS_OF)

    def test_neither_estimates_nor_surprises_raises_so_the_chain_continues(self):
        empty = {"symbol": "X", "quarterlyEarnings": []}
        with patch.object(av, "_make_api_request", _responder({
            "EARNINGS": empty,
            "EARNINGS_ESTIMATES": AlphaVantageRateLimitError("premium"),
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })):
            with self.assertRaises(NoMarketDataError):
                av.build_earnings_evidence("X", AS_OF)

    def test_estimates_returning_no_rows_is_a_named_gap(self):
        evidence = self.build({
            "EARNINGS": EARNINGS_PAYLOAD,
            "EARNINGS_ESTIMATES": {"symbol": "IBM", "estimates": []},
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })
        self.assertTrue(any("no estimate rows" in g for g in evidence.data_gaps))

    def test_only_unrecognised_horizons_is_a_named_gap(self):
        evidence = self.build({
            "EARNINGS": EARNINGS_PAYLOAD,
            "EARNINGS_ESTIMATES": {
                "symbol": "IBM",
                "estimates": [{"horizon": "decade ahead", "eps_estimate_average": "20"}],
            },
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })
        joined = " ".join(evidence.data_gaps)
        self.assertIn("does not recognise", joined)
        self.assertIn("decade ahead", joined)

    def test_the_by_design_gaps_are_always_declared(self):
        evidence = self.build({
            "EARNINGS": EARNINGS_PAYLOAD,
            "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })
        joined = " ".join(evidence.data_gaps)
        self.assertIn("Whisper expectations", joined)
        self.assertIn("margin revisions", joined)


class CalendarCsvTests(AlphaVantageTestCase):
    def test_the_earliest_future_row_for_this_symbol_is_chosen(self):
        evidence = self.build({
            "EARNINGS": EARNINGS_PAYLOAD,
            "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })
        self.assertEqual(evidence.calendar.next_date, "2026-10-22")
        self.assertAlmostEqual(evidence.calendar.eps_estimate_avg.value, 2.75)

    def test_another_symbols_rows_are_ignored(self):
        evidence = self.build({
            "EARNINGS": EARNINGS_PAYLOAD,
            "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })
        self.assertNotEqual(evidence.calendar.next_date, "2026-10-28")

    def test_the_expected_date_is_marked_estimated(self):
        evidence = self.build({
            "EARNINGS": EARNINGS_PAYLOAD,
            "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })
        self.assertTrue(evidence.calendar.date_is_estimated)

    def test_a_calendar_with_no_future_row_is_unavailable_with_a_reason(self):
        past_only = (
            "symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"
            "IBM,International Business Machines,2026-01-28,2025-12-31,3.78,USD\n"
        )
        evidence = self.build({
            "EARNINGS": EARNINGS_PAYLOAD,
            "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": past_only,
        })
        self.assertFalse(evidence.calendar.available)
        self.assertIn("no upcoming date", evidence.calendar.unavailable_reason)

    def test_a_declined_calendar_degrades_without_failing(self):
        evidence = self.build({
            "EARNINGS": EARNINGS_PAYLOAD,
            "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": AlphaVantageRateLimitError("premium"),
        })
        self.assertFalse(evidence.calendar.available)
        self.assertTrue(evidence.periods)

    def test_malformed_csv_is_treated_as_no_calendar(self):
        for raw in ("", "   ", "not,a,calendar\n1,2"):
            with self.subTest(raw=raw):
                evidence = self.build({
                    "EARNINGS": EARNINGS_PAYLOAD,
                    "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
                    "EARNINGS_CALENDAR": raw,
                })
                self.assertFalse(evidence.calendar.available)


class TranscriptTests(unittest.TestCase):
    def test_the_quarter_is_derived_from_a_reported_date_at_or_before_as_of(self):
        """Asking for the current quarter requests a call that has not happened."""
        seen = {}

        def responder(function_name, params):
            if function_name == "EARNINGS":
                return json.dumps(EARNINGS_PAYLOAD)
            seen["quarter"] = params.get("quarter")
            return json.dumps(TRANSCRIPT_PAYLOAD)

        with patch.object(av, "_make_api_request", responder):
            text = av.get_earnings_commentary("IBM", AS_OF)

        # Latest announcement at or before 2026-08-30 is 2026-07-22 for the
        # quarter ending 2026-06-30, which is 2026Q2.
        self.assertEqual(seen["quarter"], "2026Q2")
        self.assertIn("Arvind Krishna", text)
        self.assertIn("raising our full-year free cash flow guidance", text)

    def test_a_future_quarter_is_never_requested(self):
        def responder(function_name, params):
            if function_name == "EARNINGS":
                return json.dumps(EARNINGS_PAYLOAD)
            self.assertNotEqual(params.get("quarter"), "2026Q3")
            return json.dumps(TRANSCRIPT_PAYLOAD)

        with patch.object(av, "_make_api_request", responder):
            av.get_earnings_commentary("IBM", AS_OF)

    def test_fiscal_quarter_labels(self):
        cases = [
            ("2026-03-31", "2026Q1"), ("2026-06-30", "2026Q2"),
            ("2026-09-30", "2026Q3"), ("2026-12-31", "2026Q4"),
            ("2026-01-31", "2026Q1"),
        ]
        for fiscal_end, expected in cases:
            with self.subTest(fiscal_end=fiscal_end):
                self.assertEqual(av._fiscal_quarter_label(fiscal_end), expected)
        self.assertIsNone(av._fiscal_quarter_label("garbage"))

    def test_no_announced_call_yields_an_explicit_unavailable_marker(self):
        empty = {"symbol": "X", "quarterlyEarnings": []}
        with patch.object(av, "_make_api_request", _responder({"EARNINGS": empty})):
            text = av.get_earnings_commentary("X", AS_OF)
        self.assertTrue(text.startswith("EARNINGS_COMMENTARY_UNAVAILABLE"))
        self.assertIn("Do not characterise management commentary", text)

    def test_a_declined_transcript_says_it_is_an_entitlement_limit(self):
        def responder(function_name, params):
            if function_name == "EARNINGS":
                return json.dumps(EARNINGS_PAYLOAD)
            raise AlphaVantageRateLimitError("premium endpoint")

        with patch.object(av, "_make_api_request", responder):
            text = av.get_earnings_commentary("IBM", AS_OF)
        self.assertTrue(text.startswith("EARNINGS_COMMENTARY_UNAVAILABLE"))
        self.assertIn("not evidence about the company", text)

    def test_an_empty_transcript_body_is_reported_unavailable(self):
        def responder(function_name, params):
            if function_name == "EARNINGS":
                return json.dumps(EARNINGS_PAYLOAD)
            return json.dumps({"symbol": "IBM", "quarter": "2026Q2", "transcript": []})

        with patch.object(av, "_make_api_request", responder):
            text = av.get_earnings_commentary("IBM", AS_OF)
        self.assertTrue(text.startswith("EARNINGS_COMMENTARY_UNAVAILABLE"))

    def test_empty_speaker_turns_are_dropped(self):
        turns = av._parse_transcript(json.dumps(TRANSCRIPT_PAYLOAD))
        self.assertEqual(len(turns), 2)
        self.assertNotIn("Operator", " ".join(turns))

    def test_transcript_is_capped_so_one_call_cannot_flood_a_prompt(self):
        payload = {
            "symbol": "X", "quarter": "2026Q2",
            "transcript": [
                {"speaker": f"S{i}", "content": "words"} for i in range(200)
            ],
        }
        self.assertEqual(len(av._parse_transcript(json.dumps(payload), max_turns=40)), 40)

    def test_a_malformed_transcript_payload_yields_no_turns(self):
        for raw in ("not json", json.dumps([]), json.dumps({"transcript": "text"})):
            with self.subTest(raw=raw):
                self.assertEqual(av._parse_transcript(raw), [])

    def test_the_header_names_the_source_and_forbids_invention(self):
        def responder(function_name, params):
            if function_name == "EARNINGS":
                return json.dumps(EARNINGS_PAYLOAD)
            return json.dumps(TRANSCRIPT_PAYLOAD)

        with patch.object(av, "_make_api_request", responder):
            text = av.get_earnings_commentary("IBM", AS_OF)
        self.assertIn("EARNINGS_CALL_TRANSCRIPT", text)
        self.assertIn("do not summarise a call that is not here", text)


class JsonToolOutputTests(AlphaVantageTestCase):
    def test_the_routed_entry_point_returns_parseable_json(self):
        with patch.object(av, "_make_api_request", _responder({
            "EARNINGS": EARNINGS_PAYLOAD,
            "EARNINGS_ESTIMATES": ESTIMATES_PAYLOAD,
            "EARNINGS_CALENDAR": CALENDAR_CSV,
        })):
            payload = json.loads(av.get_earnings_evidence("IBM", AS_OF))
        self.assertEqual(payload["symbol"], "IBM")
        self.assertIn("Alpha Vantage EARNINGS", " ".join(payload["sources"]))


if __name__ == "__main__":
    unittest.main()

"""Deterministic earnings evidence: scoring, change arithmetic, serialization, rendering.

No network, no disk, no LLM. These tests pin the constants the plan required to be
locked rather than tuned, because moving any of them silently changes what a
published "Strong Positive" means to a reader.
"""

import json
import unittest

from tradingagents.dataflows.earnings_models import (
    BAND_NEGATIVE,
    BAND_POSITIVE,
    BAND_STRONG_NEGATIVE,
    BAND_STRONG_POSITIVE,
    MIN_AVAILABLE_WEIGHT,
    MOMENTUM_SCALES,
    MOMENTUM_WEIGHTS,
    EarningsCalendar,
    EarningsEvidence,
    EstimateTrend,
    FiscalPeriod,
    PeriodEvidence,
    RevisionBreadth,
    SurpriseEvent,
    Value,
    band_for_score,
    bounded,
    compute_momentum,
    finalize_evidence,
    period_sort_key,
    render_evidence_report,
    resolve_annual_period_end,
    safe_date,
    safe_float,
    safe_int,
    safe_ratio,
    symmetric_change,
)


def _eps(current=None, d7=None, d30=None, d60=None, d90=None, currency="USD"):
    def v(x):
        return Value(x, currency=currency) if x is not None else Value.missing("absent")
    return EstimateTrend(
        current=v(current), days_ago_7=v(d7), days_ago_30=v(d30),
        days_ago_60=v(d60), days_ago_90=v(d90),
    )


def _breadth(up7=None, down7=None, up30=None, down30=None):
    def c(x):
        return Value(x, unit="count") if x is not None else Value.missing("absent", unit="count")
    return RevisionBreadth(up_7d=c(up7), down_7d=c(down7), up_30d=c(up30), down_30d=c(down30))


def _period(key="0y", end="2026-12-31", eps=None, breadth=None, analysts=None, revenue=None):
    return PeriodEvidence(
        period=FiscalPeriod(key=key, end_date=end),
        eps=eps if eps is not None else _eps(),
        revenue=revenue if revenue is not None else _eps(currency="USD"),
        breadth=breadth if breadth is not None else _breadth(),
        analyst_count=(
            Value(analysts, unit="count") if analysts is not None
            else Value.missing("absent", unit="count")
        ),
    )


class SafeCoercionTests(unittest.TestCase):
    def test_unusable_numerics_become_none_not_zero(self):
        for raw in (None, "", "abc", float("nan"), float("inf"), float("-inf"), True, False, [], {}):
            with self.subTest(raw=raw):
                self.assertIsNone(safe_float(raw))

    def test_numeric_strings_and_floats_coerce(self):
        self.assertEqual(safe_float("8.81249"), 8.81249)
        self.assertEqual(safe_float(-2.44), -2.44)
        self.assertEqual(safe_float(0), 0.0)

    def test_zero_is_a_measurement_not_a_missing_value(self):
        # The whole missing-vs-zero distinction rests on this.
        self.assertEqual(safe_float(0.0), 0.0)
        self.assertIsNotNone(safe_float(0.0))

    def test_safe_int_rounds_and_rejects_nan(self):
        self.assertEqual(safe_int("21"), 21)
        self.assertEqual(safe_int(20.6), 21)
        self.assertIsNone(safe_int(float("nan")))

    def test_safe_date_accepts_the_shapes_providers_actually_send(self):
        import datetime as dt
        import pandas as pd

        cases = [
            (dt.date(2026, 10, 29), "2026-10-29"),
            (dt.datetime(2026, 10, 29, 16, 30), "2026-10-29"),
            (pd.Timestamp("2026-10-29"), "2026-10-29"),
            ("2026-10-29", "2026-10-29"),
            ("2026-10-29T16:30:00Z", "2026-10-29"),
            ("2026/10/29", "2026-10-29"),
            ("20261029", "2026-10-29"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(safe_date(raw), expected)

    def test_safe_date_rejects_placeholder_strings(self):
        for raw in (None, "", "  ", "nan", "NaT", "none", "null", "-", "not a date"):
            with self.subTest(raw=raw):
                self.assertIsNone(safe_date(raw))


class SymmetricChangeTests(unittest.TestCase):
    """The single most consequential function here: it decides sign."""

    def test_positive_eps_upgrade(self):
        # AAPL FY2026, live payload: 8.76760 -> 8.81249
        self.assertAlmostEqual(symmetric_change(8.81249, 8.76760), 0.005107, places=6)

    def test_narrowing_loss_is_an_upgrade_not_a_downgrade(self):
        # RIVN FY2026, live payload: -2.60537 -> -2.43642. Ordinary percentage
        # change reports -0.0648 here, inverting the meaning; Yahoo's own
        # `growth` column has the same bug.
        result = symmetric_change(-2.43642, -2.60537)
        self.assertGreater(result, 0)
        self.assertAlmostEqual(result, 0.067020, places=6)

    def test_widening_loss_is_a_downgrade(self):
        self.assertLess(symmetric_change(-3.00, -2.44), 0)

    def test_downgrade_on_positive_eps(self):
        # 600519.SS FY2026, live payload: 68.09900 -> 66.86079
        self.assertAlmostEqual(symmetric_change(66.86079, 68.09900), -0.018349, places=6)

    def test_crossing_zero_saturates_instead_of_exploding(self):
        self.assertEqual(symmetric_change(0.10, -0.10), 2.0)
        self.assertEqual(symmetric_change(-0.10, 0.10), -2.0)

    def test_old_value_of_zero_does_not_divide_by_zero(self):
        self.assertEqual(symmetric_change(1.0, 0.0), 2.0)
        self.assertEqual(symmetric_change(-1.0, 0.0), -2.0)

    def test_both_zero_is_no_change(self):
        self.assertEqual(symmetric_change(0.0, 0.0), 0.0)

    def test_both_indistinguishable_from_zero_but_unequal_is_incomparable(self):
        # The direction is real but the ratio is meaningless at that magnitude,
        # so neither 0.0 nor +/-2.0 would be honest.
        self.assertIsNone(symmetric_change(1e-10, -1e-10))

    def test_none_propagates(self):
        self.assertIsNone(symmetric_change(None, 1.0))
        self.assertIsNone(symmetric_change(1.0, None))

    def test_result_is_bounded_to_plus_minus_two(self):
        for today, old in ((1e9, -1e9), (-1e9, 1e9), (1e-6, -1e9), (5, -5)):
            with self.subTest(today=today, old=old):
                self.assertLessEqual(abs(symmetric_change(today, old)), 2.0)


class BoundedAndRatioTests(unittest.TestCase):
    def test_bounded_saturates_at_scale(self):
        self.assertEqual(bounded(0.04, 0.04), 1.0)
        self.assertEqual(bounded(0.40, 0.04), 1.0)
        self.assertEqual(bounded(-0.40, 0.04), -1.0)
        self.assertAlmostEqual(bounded(0.02, 0.04), 0.5)

    def test_bounded_rejects_non_positive_scale(self):
        with self.assertRaises(ValueError):
            bounded(0.1, 0)

    def test_safe_ratio_refuses_a_vanishing_denominator(self):
        self.assertIsNone(safe_ratio(1.0, 0.0))
        self.assertIsNone(safe_ratio(1.0, None))
        self.assertAlmostEqual(safe_ratio(13, 29), 0.448276, places=6)


class BandBoundaryTests(unittest.TestCase):
    """Boundaries are inclusive on the positive side, exclusive on the negative."""

    def test_locked_constants(self):
        self.assertEqual(BAND_STRONG_POSITIVE, 0.60)
        self.assertEqual(BAND_POSITIVE, 0.20)
        self.assertEqual(BAND_NEGATIVE, -0.20)
        self.assertEqual(BAND_STRONG_NEGATIVE, -0.60)
        self.assertEqual(MIN_AVAILABLE_WEIGHT, 0.50)
        self.assertEqual(MOMENTUM_WEIGHTS, {
            "eps_7d": 0.15, "eps_30d": 0.35, "eps_90d": 0.20,
            "breadth_30d": 0.20, "revenue_30d": 0.10,
        })
        self.assertAlmostEqual(sum(MOMENTUM_WEIGHTS.values()), 1.0)
        self.assertEqual(MOMENTUM_SCALES, {
            "eps_7d": 0.02, "eps_30d": 0.04, "eps_90d": 0.08, "revenue_30d": 0.03,
        })

    def test_every_boundary(self):
        cases = [
            (1.0, "Strong Positive"), (0.60, "Strong Positive"),
            (0.5999, "Positive"), (0.20, "Positive"),
            (0.1999, "Neutral"), (0.0, "Neutral"), (-0.1999, "Neutral"),
            (-0.20, "Negative"), (-0.5999, "Negative"),
            (-0.60, "Strong Negative"), (-1.0, "Strong Negative"),
        ]
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(band_for_score(score), expected)


class MomentumTests(unittest.TestCase):
    def test_full_coverage_reproduces_the_documented_arithmetic(self):
        # AAPL FY2026 as observed live. Recomputed by hand:
        #   eps_7d  = symmetric(8.81249, 8.80708)/0.02 = 0.030700
        #   eps_30d = symmetric(8.81249, 8.76760)/0.04 = 0.127675
        #   eps_90d = symmetric(8.81249, 8.75324)/0.08 = 0.084326
        #   breadth = (21-8)/29                        = 0.448276
        #   weight  = 0.15+0.35+0.20+0.20              = 0.90
        period = _period(
            eps=_eps(8.81249, 8.80708, 8.76760, 8.75958, 8.75324),
            breadth=_breadth(up7=20, down7=9, up30=21, down30=8),
            analysts=37,
        )
        m = compute_momentum(period)
        self.assertAlmostEqual(m.available_weight, 0.90)
        expected = (
            0.030700 * 0.15 + 0.127675 * 0.35 + 0.084326 * 0.20 + 0.448276 * 0.20
        ) / 0.90
        self.assertAlmostEqual(m.score, expected, places=5)
        self.assertEqual(m.band, "Neutral")
        self.assertNotIn("revenue_30d", m.signals)
        self.assertIn("revenue_30d", m.missing_signals)

    def test_absent_signals_do_not_vote_neutral(self):
        """A missing signal must be excluded, not counted as 0.0.

        Counting it as neutral would drag every thinly-covered name toward the
        middle band and make "Neutral" mean two different things.
        """
        # eps_30d + eps_90d = 0.55 weight, which clears the floor.
        eps_only = _period(eps=_eps(11.0, d30=10.0, d90=10.0), analysts=9)
        m = compute_momentum(eps_only)
        self.assertEqual(set(m.signals), {"eps_30d", "eps_90d"})
        # symmetric(11,10) = 0.095238; it saturates at both 0.04 and 0.08 scales.
        self.assertAlmostEqual(m.score, 1.0)
        self.assertEqual(m.band, "Strong Positive")
        # Had the three absent signals voted 0.0, the score would be 0.55.
        self.assertNotAlmostEqual(m.score, 0.55)

    def test_below_the_weight_floor_is_insufficient_data(self):
        # breadth alone is 0.20 of weight and carries no EPS direction at all.
        m = compute_momentum(_period(breadth=_breadth(up30=9, down30=1), analysts=10))
        self.assertEqual(m.band, "Insufficient Data")
        self.assertIsNone(m.score)
        self.assertEqual(m.confidence, "low")

    def test_the_weight_floor_makes_an_eps_horizon_structurally_mandatory(self):
        """Non-EPS signals cannot clear the floor on their own, by construction.

        breadth (0.20) + revenue (0.10) is 0.30, below the 0.50 floor, so a band
        can never be published from breadth and revenue alone. The explicit
        "no EPS horizon" guard in compute_momentum is therefore belt-and-braces
        today; this test is what fails first if a future weight change makes it
        reachable, which is precisely when that guard starts carrying load.
        """
        non_eps = MOMENTUM_WEIGHTS["breadth_30d"] + MOMENTUM_WEIGHTS["revenue_30d"]
        self.assertLess(non_eps, MIN_AVAILABLE_WEIGHT)
        period = _period(
            breadth=_breadth(up30=20, down30=0),
            revenue=_eps(1.2e11, d30=1.0e11),
            analysts=25,
        )
        m = compute_momentum(period)
        self.assertAlmostEqual(m.available_weight, non_eps)
        self.assertEqual(m.band, "Insufficient Data")
        self.assertIsNone(m.score)

    def test_no_data_at_all_is_insufficient_not_neutral(self):
        m = compute_momentum(_period())
        self.assertEqual(m.band, "Insufficient Data")
        self.assertEqual(m.signals, {})

    def test_negative_eps_improving_reads_positive(self):
        # RIVN FY2026 live payload.
        period = _period(
            eps=_eps(-2.43642, -2.44428, -2.60537, -2.72743, -2.70010),
            breadth=_breadth(up7=10, down7=1, up30=13, down30=0),
            analysts=14,
        )
        m = compute_momentum(period)
        self.assertEqual(m.band, "Strong Positive")
        self.assertGreater(m.score, 0.6)

    def test_thin_coverage_caps_confidence(self):
        strong = _eps(11.0, 10.9, 10.0, 9.9, 9.5)
        breadth = _breadth(up7=1, down7=0, up30=1, down30=0)
        for analysts, expected in ((None, "low"), (1, "low"), (2, "medium"), (4, "medium")):
            with self.subTest(analysts=analysts):
                m = compute_momentum(_period(eps=strong, breadth=breadth, analysts=analysts))
                self.assertEqual(m.confidence, expected)
        m = compute_momentum(_period(eps=strong, breadth=breadth, analysts=30))
        self.assertEqual(m.confidence, "high")

    def test_partial_weight_caps_confidence_at_medium(self):
        # eps_30d + eps_90d = 0.55: over the 0.50 floor, under the 0.75 ceiling.
        period = _period(eps=_eps(11.0, d30=10.5, d90=10.0), analysts=40)
        m = compute_momentum(period)
        self.assertGreaterEqual(m.available_weight, MIN_AVAILABLE_WEIGHT)
        self.assertLess(m.available_weight, 0.75)
        self.assertEqual(m.confidence, "medium")

    def test_conflicting_horizons_are_retained_and_cap_confidence(self):
        # Up sharply over 7 days, down sharply over 90.
        period = _period(
            eps=_eps(10.5, 10.0, 10.2, 11.0, 12.0),
            breadth=_breadth(up7=5, down7=1, up30=6, down30=2),
            analysts=25,
        )
        m = compute_momentum(period)
        self.assertEqual(m.confidence, "medium")
        self.assertTrue(any("disagree in direction" in d for d in m.discrepancies))

    def test_revision_count_exceeding_analyst_coverage_is_retained_not_resolved(self):
        period = _period(
            eps=_eps(11.0, 10.9, 10.5, 10.4, 10.0),
            breadth=_breadth(up7=2, down7=1, up30=30, down30=20),
            analysts=12,
        )
        m = compute_momentum(period)
        self.assertTrue(any("exceeds reported analyst" in d for d in m.discrepancies))
        # Retained, not used to reject the signal.
        self.assertNotEqual(m.band, "Insufficient Data")

    def test_impossible_window_monotonicity_is_retained(self):
        """Observed live on AAPL: 9 downgrades in 7 days against 8 in 30."""
        period = _period(
            eps=_eps(8.81249, 8.80708, 8.76760, 8.75958, 8.75324),
            breadth=_breadth(up7=20, down7=9, up30=21, down30=8),
            analysts=37,
        )
        m = compute_momentum(period)
        self.assertTrue(
            any("7-day downgrades (9) exceed 30-day (8)" in d for d in m.discrepancies),
            m.discrepancies,
        )

    def test_score_is_clamped_to_the_unit_interval(self):
        period = _period(
            eps=_eps(50.0, 1.0, 1.0, 1.0, 1.0),
            breadth=_breadth(up7=40, down7=0, up30=40, down30=0),
            revenue=_eps(9e11, d30=1e9),
            analysts=40,
        )
        m = compute_momentum(period)
        self.assertLessEqual(m.score, 1.0)
        self.assertEqual(m.band, "Strong Positive")


class FiscalPeriodTests(unittest.TestCase):
    def test_label_carries_the_period_end_when_known(self):
        self.assertEqual(
            FiscalPeriod("0y", "2027-01-31").label, "FY2027 (FYE 2027-01-31)"
        )

    def test_label_falls_back_to_the_relative_key_rather_than_inventing_a_year(self):
        self.assertEqual(
            FiscalPeriod("0y").label, "Current fiscal year (relative period 0y)"
        )
        self.assertEqual(
            FiscalPeriod("+1y").label, "Next fiscal year (relative period +1y)"
        )
        self.assertNotIn("FY", FiscalPeriod("+1y").label)

    def test_quarter_label(self):
        self.assertEqual(
            FiscalPeriod("0q", "2026-09-30").label, "Quarter ending 2026-09-30"
        )

    def test_annual_detection(self):
        self.assertTrue(FiscalPeriod("0y").is_annual)
        self.assertTrue(FiscalPeriod("+2y").is_annual)
        self.assertFalse(FiscalPeriod("0q").is_annual)

    def test_resolve_annual_period_end(self):
        self.assertEqual(
            resolve_annual_period_end("0y", next_fiscal_year_end="2026-09-27"),
            "2026-09-27",
        )
        self.assertEqual(
            resolve_annual_period_end("+1y", next_fiscal_year_end="2026-09-27"),
            "2027-09-27",
        )

    def test_resolve_returns_none_without_metadata(self):
        self.assertIsNone(resolve_annual_period_end("0y", next_fiscal_year_end=None))
        self.assertIsNone(resolve_annual_period_end("0q", next_fiscal_year_end="2026-09-27"))

    def test_leap_day_fiscal_year_end_does_not_raise(self):
        self.assertEqual(
            resolve_annual_period_end("+1y", next_fiscal_year_end="2028-02-29"),
            "2029-02-28",
        )

    def test_period_sort_is_chronological_not_lexicographic(self):
        keys = ["+1y", "0y", "+2y", "0q", "+1q"]
        self.assertEqual(
            sorted(keys, key=period_sort_key), ["0q", "+1q", "0y", "+1y", "+2y"]
        )
        # Plain string sort gets this wrong, which is the bug being guarded.
        self.assertNotEqual(sorted(keys), sorted(keys, key=period_sort_key))


class ValueTests(unittest.TestCase):
    def test_availability_is_derived_and_cannot_disagree_with_the_value(self):
        self.assertTrue(Value(1.0).available)
        self.assertTrue(Value(0.0).available, "zero is a measurement")
        self.assertFalse(Value.missing("nope").available)

    def test_missing_carries_a_reason(self):
        v = Value.missing("Yahoo publishes no 90-day counts", unit="count")
        self.assertIsNone(v.value)
        self.assertEqual(v.unit, "count")
        self.assertIn("90-day", v.unavailable_reason)

    def test_round_trip_preserves_provenance(self):
        v = Value(8.81, unit="number", currency="USD", source="yfinance", as_of="2026-08-30")
        back = Value.from_dict(v.to_dict())
        self.assertEqual(back, v)

    def test_round_trip_preserves_an_absence_and_its_reason(self):
        v = Value.missing("premium endpoint", unit="count", source="Alpha Vantage")
        back = Value.from_dict(v.to_dict())
        self.assertFalse(back.available)
        self.assertEqual(back.unavailable_reason, "premium endpoint")

    def test_from_dict_tolerates_a_malformed_payload(self):
        self.assertFalse(Value.from_dict("not a dict").available)
        self.assertFalse(Value.from_dict(None).available)


class SerializationTests(unittest.TestCase):
    def _evidence(self):
        return finalize_evidence(EarningsEvidence(
            symbol="AAPL", as_of="2026-08-30", canonical_symbol="AAPL",
            company_name="Apple Inc.", currency="USD", quote_type="EQUITY",
            periods={
                "0y": _period(
                    eps=_eps(8.81249, 8.80708, 8.76760, 8.75958, 8.75324),
                    breadth=_breadth(up7=20, down7=9, up30=21, down30=8),
                    analysts=37, end="2026-09-27",
                ),
                "+1y": _period(key="+1y", end="2027-09-27", eps=_eps(9.53, d30=9.71)),
            },
            calendar=EarningsCalendar(next_date="2026-10-29", eps_estimate_avg=Value(1.98)),
            surprises=[SurpriseEvent(
                fiscal_period_end="2026-06-30",
                eps_actual=Value(2.02), eps_estimate=Value(1.89243),
                eps_difference=Value(0.13), surprise_pct=Value(0.0674, unit="pct_dec"),
            )],
            sources=["yfinance"],
        ))

    def test_json_round_trip_is_lossless_for_the_fields_the_report_uses(self):
        original = self._evidence()
        payload = json.loads(json.dumps(original.to_dict(), sort_keys=True))
        back = EarningsEvidence.from_dict(payload)
        self.assertEqual(back.symbol, original.symbol)
        self.assertEqual(back.as_of, original.as_of)
        self.assertEqual(back.status, original.status)
        self.assertEqual(back.currency, original.currency)
        self.assertEqual(sorted(back.periods), sorted(original.periods))
        self.assertEqual(back.momentum.band, original.momentum.band)
        self.assertAlmostEqual(back.momentum.score, original.momentum.score)
        self.assertEqual(back.momentum.confidence, original.momentum.confidence)
        self.assertEqual(
            back.periods["0y"].eps.current.value, original.periods["0y"].eps.current.value
        )
        self.assertEqual(
            back.periods["0y"].breadth.down_7d.value,
            original.periods["0y"].breadth.down_7d.value,
        )
        self.assertEqual(back.calendar.next_date, "2026-10-29")
        self.assertEqual(len(back.surprises), 1)
        self.assertEqual(back.data_gaps, original.data_gaps)

    def test_serialization_is_stable_across_input_ordering(self):
        a = self._evidence()
        reordered = EarningsEvidence(
            **{**a.__dict__, "periods": dict(reversed(list(a.periods.items())))}
        )
        self.assertEqual(
            json.dumps(a.to_dict(), sort_keys=True),
            json.dumps(reordered.to_dict(), sort_keys=True),
        )

    def test_nulls_are_explicit_rather_than_omitted(self):
        payload = self._evidence().to_dict()
        cell = payload["periods"]["0y"]["breadth"]["up_90d"]
        self.assertIsNone(cell["value"])
        self.assertFalse(cell["available"])
        self.assertIsNotNone(cell["unavailable_reason"])

    def test_from_dict_rejects_a_non_object_payload(self):
        with self.assertRaises(ValueError):
            EarningsEvidence.from_dict("[]")

    def test_from_dict_falls_back_to_safe_values_on_unknown_enums(self):
        back = EarningsEvidence.from_dict(
            {"symbol": "X", "as_of": "2026-01-01", "status": "bogus",
             "momentum": {"band": "Amazing", "confidence": "certain"}}
        )
        self.assertEqual(back.status, "partial")
        self.assertEqual(back.momentum.band, "Insufficient Data")
        self.assertEqual(back.momentum.confidence, "low")


class TerminalStatusTests(unittest.TestCase):
    def test_factories_carry_a_reason_and_no_figures(self):
        cases = [
            (EarningsEvidence.unsupported("SPY", "2026-08-30", "ETF"), "unsupported"),
            (EarningsEvidence.pit_unavailable("AAPL", "2020-01-01", "no vintage"), "pit_unavailable"),
            (EarningsEvidence.no_coverage("XYZ", "2026-08-30", "no analysts"), "no_coverage"),
        ]
        for evidence, expected in cases:
            with self.subTest(status=expected):
                self.assertEqual(evidence.status, expected)
                self.assertEqual(evidence.periods, {})
                self.assertEqual(evidence.momentum.band, "Insufficient Data")
                self.assertTrue(evidence.data_gaps)
                self.assertTrue(evidence.status_detail)

    def test_terminal_report_refuses_to_show_figures_and_says_so(self):
        report = render_evidence_report(
            EarningsEvidence.unsupported(
                "SPY", "2026-08-30", "SPY is an etf, not an operating company."
            )
        )
        self.assertIn("not applicable", report)
        self.assertIn("etf", report)
        self.assertIn("Do not substitute values", report)
        for forbidden in ("EPS Consensus", "Revision breadth", "Earnings Momentum"):
            self.assertNotIn(forbidden, report)


class RenderTests(unittest.TestCase):
    def _report(self):
        return render_evidence_report(finalize_evidence(EarningsEvidence(
            symbol="AAPL", as_of="2026-08-30", company_name="Apple Inc.", currency="USD",
            periods={"0y": _period(
                eps=_eps(8.81249, 8.80708, 8.76760, 8.75958, 8.75324),
                breadth=_breadth(up7=20, down7=9, up30=21, down30=8),
                analysts=37, end="2026-09-27",
            )},
            calendar=EarningsCalendar(
                next_date="2026-10-29", eps_estimate_avg=Value(1.98013, currency="USD")
            ),
            sources=["yfinance"],
        )))

    def test_headline_shows_consensus_today_against_each_lookback(self):
        report = self._report()
        self.assertIn("FY2026 (FYE 2026-09-27) EPS Consensus", report)
        self.assertIn("| 30 days ago | 8.77 USD |", report)
        self.assertIn("| **Today** | **8.81 USD** |", report)

    def test_headline_shows_revision_breadth_both_ways(self):
        report = self._report()
        self.assertIn("Last 30d: +21 raised / -8 lowered", report)
        self.assertIn("Analysts covering this period: 37", report)

    def test_ninety_day_breadth_renders_unavailable_with_its_reason(self):
        report = self._report()
        self.assertIn("Last 90d: unavailable", report)

    def test_momentum_band_and_reproducible_arithmetic_are_shown(self):
        report = self._report()
        self.assertIn("## Earnings Momentum", report)
        self.assertIn("**Neutral**", report)
        self.assertIn("| Signal | Value | Weight |", report)
        self.assertIn("eps_30d", report)

    def test_symmetric_change_is_labelled_as_such(self):
        self.assertIn("symmetric", self._report())
        self.assertIn("not an ordinary percentage", self._report())

    def test_absent_sections_state_their_absence_rather_than_disappearing(self):
        report = self._report()
        self.assertIn("## Surprise History", report)
        self.assertIn("## Post-Earnings Drift", report)
        self.assertIn("Unavailable", report)

    def test_unconfirmed_calendar_window_is_flagged(self):
        report = render_evidence_report(finalize_evidence(EarningsEvidence(
            symbol="X", as_of="2026-08-30",
            periods={"0y": _period(eps=_eps(1.0, d30=0.9), analysts=5)},
            calendar=EarningsCalendar(
                next_date="2026-11-03", next_date_range_end="2026-11-07",
                date_is_estimated=True,
            ),
        )))
        self.assertIn("2026-11-03 — 2026-11-07", report)
        self.assertIn("Unconfirmed", report)
        self.assertIn("whole window", report)

    def test_insufficient_data_is_explained_as_coverage_not_verdict(self):
        report = render_evidence_report(finalize_evidence(EarningsEvidence(
            symbol="X", as_of="2026-08-30", periods={"0y": _period()},
        )))
        self.assertIn("Insufficient Data", report)
        self.assertIn("statement about data coverage, not a neutral verdict", report)

    def test_periods_table_is_chronological(self):
        report = render_evidence_report(finalize_evidence(EarningsEvidence(
            symbol="X", as_of="2026-08-30",
            periods={
                "+1y": _period(key="+1y", end="2027-12-31", eps=_eps(2.0)),
                "0y": _period(key="0y", end="2026-12-31", eps=_eps(1.0)),
                "0q": _period(key="0q", end="2026-09-30", eps=_eps(0.5)),
            },
        )))
        table = report.split("## All Forecast Periods")[1]
        self.assertLess(table.index("2026-09-30"), table.index("2026-12-31"))
        self.assertLess(table.index("2026-12-31"), table.index("2027-12-31"))


class FinalizeTests(unittest.TestCase):
    def test_momentum_is_recomputed_from_the_periods_present(self):
        evidence = EarningsEvidence(
            symbol="X", as_of="2026-08-30",
            periods={"0y": _period(
                eps=_eps(-2.43642, -2.44428, -2.60537, -2.72743, -2.70010),
                breadth=_breadth(up7=10, down7=1, up30=13, down30=0), analysts=14,
            )},
        )
        self.assertEqual(evidence.momentum.band, "Insufficient Data")  # not yet scored
        self.assertEqual(finalize_evidence(evidence).momentum.band, "Strong Positive")

    def test_missing_signals_become_named_data_gaps(self):
        evidence = finalize_evidence(EarningsEvidence(
            symbol="X", as_of="2026-08-30",
            periods={"0y": _period(eps=_eps(1.0, d30=0.9), analysts=5)},
        ))
        joined = " ".join(evidence.data_gaps)
        self.assertIn("revenue consensus trend unavailable", joined)
        self.assertIn("revision breadth", joined)

    def test_partial_status_when_gaps_remain(self):
        evidence = finalize_evidence(EarningsEvidence(
            symbol="X", as_of="2026-08-30",
            periods={"0y": _period(eps=_eps(1.0, d30=0.9), analysts=5)},
        ))
        self.assertEqual(evidence.status, "partial")

    def test_terminal_statuses_survive_finalize(self):
        for factory in (
            EarningsEvidence.unsupported,
            EarningsEvidence.pit_unavailable,
            EarningsEvidence.no_coverage,
        ):
            with self.subTest(factory=factory.__name__):
                original = factory("X", "2026-08-30", "reason")
                self.assertEqual(finalize_evidence(original).status, original.status)

    def test_falls_back_to_the_first_annual_period_when_zero_y_is_absent(self):
        evidence = finalize_evidence(EarningsEvidence(
            symbol="X", as_of="2026-08-30",
            periods={"+1y": _period(
                key="+1y", end="2027-12-31",
                eps=_eps(11.0, 10.9, 10.0, 9.9, 9.5),
                breadth=_breadth(up7=8, down7=1, up30=9, down30=1), analysts=20,
            )},
        ))
        self.assertEqual(evidence.momentum.period_key, "+1y")
        self.assertNotEqual(evidence.momentum.band, "Insufficient Data")


if __name__ == "__main__":
    unittest.main()

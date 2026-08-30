"""Deterministic quality evidence: scoring, serialization, rendering.

No network, no disk, no LLM. Pins the constants the design required to be
locked rather than tuned in place — moving any of them silently changes what
a published "High Quality" means to a reader.
"""

import unittest

from tradingagents.dataflows.quality_models import (
    BAND_ABOVE_AVERAGE,
    BAND_BELOW_AVERAGE,
    BAND_HIGH_QUALITY,
    BAND_WEAK,
    MIN_AVAILABLE_WEIGHT,
    MIN_CONSISTENCY_PERIODS,
    QUALITY_WEIGHTS,
    QualityEvidence,
    QualityTierAssessment,
    Value,
    band_for_score,
    compute_quality_tier,
    finalize_evidence,
    render_quality_report,
)


class BandBoundaryTests(unittest.TestCase):
    """Asymmetric >=/> convention, same as earnings_models.band_for_score."""

    def test_high_quality_boundary_is_inclusive(self):
        self.assertEqual(band_for_score(BAND_HIGH_QUALITY), "High Quality")
        self.assertEqual(band_for_score(BAND_HIGH_QUALITY - 0.001), "Above Average")

    def test_above_average_boundary_is_inclusive(self):
        self.assertEqual(band_for_score(BAND_ABOVE_AVERAGE), "Above Average")
        self.assertEqual(band_for_score(BAND_ABOVE_AVERAGE - 0.001), "Average")

    def test_below_average_boundary_belongs_to_below_average(self):
        # Exclusive on the "Average" side, same asymmetric convention
        # earnings_models.band_for_score documents: the boundary value itself
        # goes to the lower band, and only strictly above it counts as Average.
        self.assertEqual(band_for_score(BAND_BELOW_AVERAGE), "Below Average")
        self.assertEqual(band_for_score(BAND_BELOW_AVERAGE + 0.001), "Average")

    def test_weak_boundary_belongs_to_weak(self):
        self.assertEqual(band_for_score(BAND_WEAK), "Weak")
        self.assertEqual(band_for_score(BAND_WEAK + 0.001), "Below Average")

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(QUALITY_WEIGHTS.values()), 1.0, places=9)


class ComputeQualityTierTests(unittest.TestCase):
    def test_all_signals_present_and_excellent_scores_high_quality(self):
        result = compute_quality_tier(
            roe=0.40, operating_margin=0.35, debt_to_equity=0.2,
            current_ratio=2.5, fcf_margin=0.20,
            margin_history=[0.34, 0.35, 0.36, 0.35],
        )
        self.assertEqual(result.tier, "High Quality")
        self.assertAlmostEqual(result.available_weight, 1.0, places=9)

    def test_all_signals_present_and_terrible_scores_weak(self):
        result = compute_quality_tier(
            roe=-0.10, operating_margin=-0.05, debt_to_equity=3.0,
            current_ratio=0.5, fcf_margin=-0.10,
            margin_history=[0.10, -0.05, 0.20, -0.15],
        )
        self.assertEqual(result.tier, "Weak")

    def test_no_signals_is_insufficient_data(self):
        result = compute_quality_tier(
            roe=None, operating_margin=None, debt_to_equity=None,
            current_ratio=None, fcf_margin=None, margin_history=None,
        )
        self.assertEqual(result.tier, "Insufficient Data")
        self.assertIsNone(result.score)
        self.assertEqual(result.available_weight, 0.0)

    def test_below_weight_floor_is_insufficient_data(self):
        # Only current_ratio (0.15 weight) available -- below the 0.50 floor.
        result = compute_quality_tier(
            roe=None, operating_margin=None, debt_to_equity=None,
            current_ratio=2.5, fcf_margin=None, margin_history=None,
        )
        self.assertLess(0.15, MIN_AVAILABLE_WEIGHT)
        self.assertEqual(result.tier, "Insufficient Data")

    def test_missing_signals_are_renormalized_not_penalized(self):
        # roe+operating_margin+debt_to_equity = 0.25+0.20+0.15 = 0.60, clearing
        # the 0.50 floor even with current_ratio/fcf_margin/consistency absent.
        result = compute_quality_tier(
            roe=0.30, operating_margin=0.30, debt_to_equity=0.5,
            current_ratio=None, fcf_margin=None, margin_history=None,
        )
        self.assertIsNotNone(result.score)
        self.assertEqual(sorted(result.missing_signals),
                         ["current_ratio", "fcf_margin", "margin_consistency"])

    def test_margin_consistency_needs_the_minimum_period_count(self):
        result = compute_quality_tier(
            roe=0.20, operating_margin=0.20, debt_to_equity=0.5,
            current_ratio=2.0, fcf_margin=0.10,
            margin_history=[0.20, 0.21][: MIN_CONSISTENCY_PERIODS - 1],
        )
        self.assertIn("margin_consistency", result.missing_signals)

    def test_low_dispersion_margin_history_scores_positively(self):
        with_history = compute_quality_tier(
            roe=0.20, operating_margin=0.20, debt_to_equity=0.5,
            current_ratio=2.0, fcf_margin=0.10,
            margin_history=[0.20, 0.201, 0.199, 0.2005],
        )
        self.assertGreater(with_history.signals["margin_consistency"], 0.5)

    def test_high_dispersion_margin_history_scores_negatively(self):
        result = compute_quality_tier(
            roe=0.20, operating_margin=0.20, debt_to_equity=0.5,
            current_ratio=2.0, fcf_margin=0.10,
            margin_history=[0.05, 0.30, 0.02, 0.28],
        )
        self.assertLess(result.signals["margin_consistency"], 0.0)

    def test_score_is_reproducible_from_published_signals_and_weights(self):
        result = compute_quality_tier(
            roe=0.20, operating_margin=0.10, debt_to_equity=1.0,
            current_ratio=1.5, fcf_margin=0.05, margin_history=None,
        )
        recomputed = sum(
            result.signals[k] * result.weights_used[k] for k in result.signals
        ) / result.available_weight
        self.assertAlmostEqual(result.score, recomputed, places=9)


class QualityTierAssessmentSerializationTests(unittest.TestCase):
    def test_round_trip(self):
        original = QualityTierAssessment(
            tier="Above Average", score=0.35, signals={"roe": 0.5},
            weights_used={"roe": 0.25}, available_weight=0.25,
            missing_signals=["current_ratio"],
        )
        restored = QualityTierAssessment.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_malformed_input_yields_insufficient_data_not_a_raise(self):
        for raw in (None, "oops", [], {"tier": "not a real tier"}):
            with self.subTest(raw=raw):
                restored = QualityTierAssessment.from_dict(raw)
                self.assertEqual(restored.tier, "Insufficient Data")


class QualityEvidenceTests(unittest.TestCase):
    def _evidence(self, **overrides):
        base = dict(
            symbol="AAPL", as_of="2026-08-30", company_name="Apple Inc.",
            currency="USD",
            return_on_equity=Value(1.4875, unit="pct_dec"),
            operating_margin=Value(0.326, unit="pct_dec"),
            profit_margin=Value(0.276, unit="pct_dec"),
            return_on_assets=Value(0.271, unit="pct_dec"),
            debt_to_equity=Value(0.784, unit="ratio"),
            current_ratio=Value(1.003, unit="ratio"),
            free_cash_flow=Value(107721875456, unit="currency_large", currency="USD"),
            total_revenue=Value(466822987776, unit="currency_large", currency="USD"),
            margin_history=[
                Value(0.326, unit="pct_dec"), Value(0.315, unit="pct_dec"),
                Value(0.302, unit="pct_dec"),
            ],
            margin_history_periods=["2025-09-30", "2024-09-30", "2023-09-30"],
            sources=["yfinance (Yahoo Finance fundamentals)"],
        )
        base.update(overrides)
        return finalize_evidence(QualityEvidence(**base))

    def test_fcf_margin_is_derived_not_stored(self):
        evidence = self._evidence()
        self.assertAlmostEqual(evidence.fcf_margin, 107721875456 / 466822987776, places=6)

    def test_finalize_computes_the_tier_from_the_fields(self):
        evidence = self._evidence()
        self.assertNotEqual(evidence.tier.tier, "Insufficient Data")
        self.assertEqual(evidence.status, "ok")

    def test_to_dict_from_dict_round_trip(self):
        evidence = self._evidence()
        restored = QualityEvidence.from_dict(evidence.to_dict())
        self.assertEqual(restored.to_dict(), evidence.to_dict())

    def test_unsupported_short_circuits_status(self):
        evidence = QualityEvidence.unsupported("SPY", "2026-08-30", "SPY is an etf.")
        self.assertEqual(evidence.status, "unsupported")
        self.assertEqual(evidence.tier.tier, "Insufficient Data")

    def test_partial_status_when_gaps_remain_despite_a_tier(self):
        evidence = self._evidence(current_ratio=Value.missing("not reported"))
        self.assertEqual(evidence.status, "partial")
        self.assertTrue(any("Current ratio" in g for g in evidence.data_gaps))


class RenderQualityReportTests(unittest.TestCase):
    def test_tier_and_signals_appear(self):
        evidence = QualityEvidenceTests()._evidence()
        report = render_quality_report(evidence)
        self.assertIn("## Quality Tier", report)
        self.assertIn(evidence.tier.tier, report)
        self.assertIn("| roe |", report)

    def test_terminal_status_report_has_no_tier_section(self):
        evidence = QualityEvidence.no_coverage("XYZ", "2026-08-30", "No coverage.")
        report = render_quality_report(evidence)
        self.assertIn("No fundamentals coverage", report)
        self.assertNotIn("## Quality Tier", report)

    def test_insufficient_data_explains_itself(self):
        evidence = finalize_evidence(QualityEvidence(symbol="XYZ", as_of="2026-08-30"))
        report = render_quality_report(evidence)
        self.assertIn("not a neutral verdict on the business", report)

    def test_margin_history_table_renders_when_present(self):
        evidence = QualityEvidenceTests()._evidence(
            margin_history=[Value(0.30, unit="pct_dec"), Value(0.32, unit="pct_dec"),
                           Value(0.31, unit="pct_dec")],
            margin_history_periods=["2025-09-30", "2024-09-30", "2023-09-30"],
        )
        report = render_quality_report(evidence)
        self.assertIn("## Operating Margin History", report)
        self.assertIn("2025-09-30", report)


if __name__ == "__main__":
    unittest.main()

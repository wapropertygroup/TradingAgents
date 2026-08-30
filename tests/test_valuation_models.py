"""Deterministic valuation evidence: scoring, serialization, rendering.

No network, no disk, no LLM. Same discipline as ``test_quality_models.py``:
pins the locked constants so a threshold move is a deliberate, visible change.
"""

import unittest

from tradingagents.dataflows.valuation_models import (
    BAND_ATTRACTIVE,
    BAND_DEEP_VALUE,
    BAND_EXPENSIVE,
    BAND_EXTREME_PREMIUM,
    MIN_AVAILABLE_WEIGHT,
    VALUATION_WEIGHTS,
    ValuationEvidence,
    ValuationTierAssessment,
    Value,
    band_for_score,
    compute_valuation_tier,
    finalize_evidence,
    render_valuation_report,
)


class BandBoundaryTests(unittest.TestCase):
    def test_deep_value_boundary_is_inclusive(self):
        self.assertEqual(band_for_score(BAND_DEEP_VALUE), "Deep Value")
        self.assertEqual(band_for_score(BAND_DEEP_VALUE - 0.001), "Attractive")

    def test_attractive_boundary_is_inclusive(self):
        self.assertEqual(band_for_score(BAND_ATTRACTIVE), "Attractive")
        self.assertEqual(band_for_score(BAND_ATTRACTIVE - 0.001), "Fair")

    def test_expensive_boundary_belongs_to_expensive(self):
        self.assertEqual(band_for_score(BAND_EXPENSIVE), "Expensive")
        self.assertEqual(band_for_score(BAND_EXPENSIVE + 0.001), "Fair")

    def test_extreme_premium_boundary_belongs_to_extreme_premium(self):
        self.assertEqual(band_for_score(BAND_EXTREME_PREMIUM), "Extreme Premium")
        self.assertEqual(band_for_score(BAND_EXTREME_PREMIUM + 0.001), "Expensive")

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(VALUATION_WEIGHTS.values()), 1.0, places=9)


class ComputeValuationTierTests(unittest.TestCase):
    def test_cheap_across_every_signal_scores_deep_value(self):
        result = compute_valuation_tier(
            trailing_pe=8.0, forward_pe=6.0, peg_ratio=0.4,
            price_to_book=1.0, dividend_yield_pct=0.05,
        )
        self.assertEqual(result.tier, "Deep Value")

    def test_expensive_across_every_signal_scores_extreme_premium(self):
        result = compute_valuation_tier(
            trailing_pe=45.0, forward_pe=50.0, peg_ratio=4.0,
            price_to_book=8.0, dividend_yield_pct=0.0,
        )
        self.assertEqual(result.tier, "Extreme Premium")

    def test_no_signals_is_insufficient_data(self):
        result = compute_valuation_tier(
            trailing_pe=None, forward_pe=None, peg_ratio=None,
            price_to_book=None, dividend_yield_pct=None,
        )
        self.assertEqual(result.tier, "Insufficient Data")
        self.assertIsNone(result.score)

    def test_negative_earnings_growth_stock_can_still_score_via_peg_and_pb(self):
        # NVDA-like case: headline P/E alone looks expensive, but PEG (growth
        # priced in) and a big forward/trailing spread pull the tier back.
        # This is the whole point of a weighted blend over a single P/E cutoff.
        result = compute_valuation_tier(
            trailing_pe=28.8, forward_pe=14.2, peg_ratio=0.63,
            price_to_book=27.0, dividend_yield_pct=0.0044,
        )
        self.assertIn(result.tier, ("Fair", "Attractive"))
        self.assertLess(result.signals["pe_band"], 0)
        self.assertGreater(result.signals["peg"], 0)
        self.assertGreater(result.signals["forward_vs_trailing"], 0)

    def test_below_weight_floor_is_insufficient_data(self):
        # Only dividend_yield (0.10 weight) available -- below the 0.50 floor.
        result = compute_valuation_tier(
            trailing_pe=None, forward_pe=None, peg_ratio=None,
            price_to_book=None, dividend_yield_pct=0.02,
        )
        self.assertLess(0.10, MIN_AVAILABLE_WEIGHT)
        self.assertEqual(result.tier, "Insufficient Data")

    def test_zero_dividend_is_neutral_not_penalized(self):
        result = compute_valuation_tier(
            trailing_pe=20.0, forward_pe=20.0, peg_ratio=1.5,
            price_to_book=3.0, dividend_yield_pct=0.0,
        )
        self.assertEqual(result.signals["dividend_yield"], 0.0)

    def test_forward_cheaper_than_trailing_is_positive(self):
        result = compute_valuation_tier(
            trailing_pe=30.0, forward_pe=20.0, peg_ratio=None,
            price_to_book=None, dividend_yield_pct=None,
        )
        self.assertGreater(result.signals["forward_vs_trailing"], 0)

    def test_forward_more_expensive_than_trailing_is_negative(self):
        result = compute_valuation_tier(
            trailing_pe=15.0, forward_pe=20.0, peg_ratio=None,
            price_to_book=None, dividend_yield_pct=None,
        )
        self.assertLess(result.signals["forward_vs_trailing"], 0)

    def test_score_is_reproducible_from_published_signals_and_weights(self):
        result = compute_valuation_tier(
            trailing_pe=18.0, forward_pe=16.0, peg_ratio=1.2,
            price_to_book=2.5, dividend_yield_pct=0.015,
        )
        recomputed = sum(
            result.signals[k] * result.weights_used[k] for k in result.signals
        ) / result.available_weight
        self.assertAlmostEqual(result.score, recomputed, places=9)


class ValuationTierAssessmentSerializationTests(unittest.TestCase):
    def test_round_trip(self):
        original = ValuationTierAssessment(
            tier="Fair", score=-0.05, signals={"pe_band": -0.1},
            weights_used={"pe_band": 0.30}, available_weight=0.30,
            missing_signals=["peg"],
        )
        restored = ValuationTierAssessment.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_malformed_input_yields_insufficient_data_not_a_raise(self):
        for raw in (None, "oops", [], {"tier": "not a real tier"}):
            with self.subTest(raw=raw):
                restored = ValuationTierAssessment.from_dict(raw)
                self.assertEqual(restored.tier, "Insufficient Data")


class ValuationEvidenceTests(unittest.TestCase):
    def _evidence(self, **overrides):
        base = dict(
            symbol="AAPL", as_of="2026-08-30", company_name="Apple Inc.",
            currency="USD",
            trailing_pe=Value(36.62, unit="ratio"),
            forward_pe=Value(33.52, unit="ratio"),
            peg_ratio=Value(2.54, unit="ratio"),
            price_to_book=Value(43.44, unit="ratio"),
            dividend_yield=Value(0.0034, unit="pct_dec"),
            market_cap=Value(4.67e12, unit="currency_large", currency="USD"),
            sources=["yfinance (Yahoo Finance fundamentals)"],
        )
        base.update(overrides)
        return finalize_evidence(ValuationEvidence(**base))

    def test_finalize_computes_the_tier_from_the_fields(self):
        evidence = self._evidence()
        self.assertNotEqual(evidence.tier.tier, "Insufficient Data")
        self.assertEqual(evidence.status, "ok")

    def test_to_dict_from_dict_round_trip(self):
        evidence = self._evidence()
        restored = ValuationEvidence.from_dict(evidence.to_dict())
        self.assertEqual(restored.to_dict(), evidence.to_dict())

    def test_unsupported_short_circuits_status(self):
        evidence = ValuationEvidence.unsupported("SPY", "2026-08-30", "SPY is an etf.")
        self.assertEqual(evidence.status, "unsupported")
        self.assertEqual(evidence.tier.tier, "Insufficient Data")

    def test_partial_status_when_gaps_remain_despite_a_tier(self):
        evidence = self._evidence(peg_ratio=Value.missing("not reported"))
        self.assertEqual(evidence.status, "partial")
        self.assertTrue(any("PEG" in g for g in evidence.data_gaps))


class RenderValuationReportTests(unittest.TestCase):
    def test_tier_and_signals_appear(self):
        evidence = ValuationEvidenceTests()._evidence()
        report = render_valuation_report(evidence)
        self.assertIn("## Valuation Tier", report)
        self.assertIn(evidence.tier.tier, report)
        self.assertIn("| pe_band |", report)

    def test_terminal_status_report_has_no_tier_section(self):
        evidence = ValuationEvidence.no_coverage("XYZ", "2026-08-30", "No coverage.")
        report = render_valuation_report(evidence)
        self.assertIn("No valuation coverage", report)
        self.assertNotIn("## Valuation Tier", report)

    def test_insufficient_data_mentions_negative_earnings_as_the_likely_reason(self):
        evidence = finalize_evidence(ValuationEvidence(symbol="XYZ", as_of="2026-08-30"))
        report = render_valuation_report(evidence)
        self.assertIn("negative-earnings company", report)


if __name__ == "__main__":
    unittest.main()

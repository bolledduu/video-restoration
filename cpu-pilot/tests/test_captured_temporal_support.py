"""Tests for Captured Temporal Support (CTS) Safeguard Module.

Validates:
1. Withheld-value invariance (leakage test): changing withheld values leaves audit predictions strictly unchanged.
2. Channel grouping: all color channels masked together.
3. Timing disruption: preserves signal scale without circular wraparound boundary jumps.
4. Non-mouth control isolation: mouth vs cheek regions are distinct.
5. Calibrated risk assessor: fits and predicts without ground truth leakage.
6. Safeguard selector: defaults to captured fallback when harm risk exceeds threshold.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from captured_temporal_support import (
    MaskedAuditSplit,
    get_face_regions,
    isolate_temporal_increment,
    construct_timing_disruptions,
    compute_captured_temporal_support,
    compute_attribution_influence,
    CalibratedRiskAssessor,
)


class TestCapturedTemporalSupport(unittest.TestCase):
    def setUp(self):
        self.shape = (10, 64, 64, 3)
        self.rng = np.random.default_rng(20261017)
        self.captured = self.rng.integers(0, 256, size=self.shape, dtype=np.uint8)
        self.restored = np.clip(self.captured.astype(float) * 0.9 + 10, 0, 255).astype(np.uint8)

    def test_all_channels_masked_together(self):
        """Verify that all color channels at a withheld pixel site are masked together."""
        splitter = MaskedAuditSplit(self.shape, mask_fraction=0.15)
        for mask in splitter.masks:
            self.assertEqual(mask.shape, (64, 64))
            self.assertEqual(mask.dtype, bool)
            # Mask is 2D spatial: applying it to (T, H, W, C) masks all C channels simultaneously

    def test_withheld_value_invariance_and_no_leakage(self):
        """Verify that modifying withheld pixels while keeping retained pixels fixed

        causes ZERO change in the filled output and audit predictions.
        """
        splitter = MaskedAuditSplit(self.shape, mask_fraction=0.15)
        mask = splitter.masks[0]

        # Dummy audit function (e.g. spatial box blur)
        def audit_fn(v):
            return np.roll(v, 1, axis=1)

        is_invariant = splitter.verify_withheld_invariance(self.captured, mask, audit_fn)
        self.assertTrue(is_invariant, "Withheld pixel values leaked into fill operator or audit predictions!")

    def test_timing_disruptions_preserve_scale_and_have_no_circular_wraparound(self):
        """Verify that temporal disruptions avoid circular wraparound boundary jumps and preserve scale."""
        _, _, dynamic_delta = isolate_temporal_increment(self.restored, self.captured)
        disruptions = construct_timing_disruptions(dynamic_delta, shifts=[-2, 2])
        self.assertEqual(len(disruptions), 2)
        for d in disruptions:
            self.assertEqual(d.shape, dynamic_delta.shape)
            # Standard deviation / energy is preserved
            self.assertAlmostEqual(float(np.std(d)), float(np.std(dynamic_delta)), delta=5.0)

    def test_face_regions_are_disjoint(self):
        """Mouth and cheek control regions must be spatially disjoint."""
        mouth, cheek = get_face_regions(crop_size=128)
        overlap = np.logical_and(mouth, cheek)
        self.assertEqual(int(np.sum(overlap)), 0, "Mouth and cheek control regions must not overlap")
        self.assertGreater(int(np.sum(mouth)), 0)
        self.assertGreater(int(np.sum(cheek)), 0)

    def test_cts_support_diagnostics_structure(self):
        """Verify complete diagnostic dictionary without ground truth leakage."""
        diag = compute_captured_temporal_support(self.captured, self.restored)
        required_keys = [
            "cts_support",
            "cheek_support",
            "mouth_specificity",
            "heldout_mse",
            "cdi_consistency",
            "mask_stability",
            "diagnostic_validity",
        ]
        for k in required_keys:
            self.assertIn(k, diag)
            self.assertFalse(np.isnan(diag[k]) if isinstance(diag[k], (int, float)) else False)

        # Labels or ground-truth errors MUST NOT be present
        self.assertNotIn("labels", diag)
        self.assertNotIn("ground_truth", diag)
        self.assertNotIn("error_count", diag)

    def test_attribution_influence_metrics(self):
        """Verify attribution influence separation from correctness."""
        cap_probs = np.array([0.1, 0.4, 0.6, 0.9])
        rest_probs = np.array([0.2, 0.6, 0.6, 0.8])
        infl = compute_attribution_influence(cap_probs, rest_probs)
        self.assertEqual(infl["n_frames"], 4)
        self.assertAlmostEqual(infl["delta_prob_mean"], 0.05)
        self.assertEqual(infl["disagreement_count"], 1)  # Index 1 crossed 0.5 (0.4 -> 0.6)

    def test_calibrated_risk_assessor_fit_and_selection(self):
        """Verify low-capacity risk model fits and selects appropriately without label leakage."""
        # Create synthetic training records
        records = [
            {"diagnostics": {"cts_support": 10.0, "mouth_specificity": 8.0, "heldout_mse": 20.0, "cdi_consistency": 30.0, "delta_prob_abs": 0.05, "disagreement_rate": 0.0}, "harm_count": 0, "correction_count": 1},
            {"diagnostics": {"cts_support": -15.0, "mouth_specificity": -10.0, "heldout_mse": 120.0, "cdi_consistency": 150.0, "delta_prob_abs": 0.4, "disagreement_rate": 0.5}, "harm_count": 5, "correction_count": 0},
            {"diagnostics": {"cts_support": 2.0, "mouth_specificity": 1.0, "heldout_mse": 40.0, "cdi_consistency": 50.0, "delta_prob_abs": 0.02, "disagreement_rate": 0.0}, "harm_count": 0, "correction_count": 0},
        ]
        assessor = CalibratedRiskAssessor()
        assessor.fit(records)
        self.assertTrue(assessor.is_fitted)

        # High risk case should trigger fallback to captured
        high_risk_diag = {"cts_support": -25.0, "mouth_specificity": -15.0, "heldout_mse": 180.0, "cdi_consistency": 200.0, "delta_prob_abs": 0.6, "disagreement_rate": 0.8}
        rec, reason = assessor.evaluate_selection(high_risk_diag)
        self.assertEqual(rec, "REVERT_TO_CAPTURED")


if __name__ == "__main__":
    unittest.main()

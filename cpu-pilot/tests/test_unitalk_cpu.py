import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_unitalk_cpu import event_mask, selective
from run_unitalk_cpu import degrade


class UnitalkCPUTests(unittest.TestCase):
    def test_joint_event_requires_quality_error_and_causality(self):
        y = np.array([0, 0, 0, 1])
        ref = np.array([.1, .1, .1, .9])
        match = np.array([.1, .1, .7, .9])
        swap = np.array([.8, .8, .9, .1])
        result = event_mask(y, ref, ref, match, swap, np.array([.6, .4, .6, .6]), np.full(4, .5))
        self.assertEqual(result.tolist(), [True, False, False, True])

    def test_safeguards_have_equal_nonzero_coverage(self):
        records = [dict(id=str(i), label=0, p=p, conditioning_sensitivity=p,
                        capture_disagreement=p, reprojection_residual=p, confidence=-abs(p-.5))
                   for i, p in enumerate([.95, .9, .1, .2])]
        result = selective(records, .5)
        self.assertEqual(result['conditioning_sensitivity']['coverage'], .5)
        self.assertEqual(result['confidence']['coverage'], .5)
        self.assertEqual(result['conditioning_sensitivity']['error_rate'], 0)
        self.assertGreaterEqual(result['confidence']['error_rate'], .5)

    def test_degradation_is_deterministic_and_keeps_fixed_geometry(self):
        frame = np.arange(128 * 128 * 3, dtype=np.uint8).reshape(128, 128, 3)
        for level in (4, 8):
            small, display = degrade(frame, level)
            self.assertEqual(small.shape, (8, 8, 3))
            self.assertEqual(display.shape, frame.shape)
            self.assertTrue(np.array_equal(degrade(frame, level)[0], small))


if __name__ == '__main__':
    unittest.main()

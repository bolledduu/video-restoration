import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from wvaq_quality_pilot import choose, selection


class QualitySelectionTests(unittest.TestCase):
    def candidates(self):
        return {str(i): dict(brisque=4-i, error=i/4, visual_error=i/4,
            brier=i/4, ssim=.5, psnr=20, confidence=.6) for i in range(4)}

    def test_blind_choice_is_independent_of_labels_and_reference(self):
        values = self.candidates()
        before = choose(values, list(values), 'brisque')
        for i, candidate in enumerate(values.values()):
            candidate.update(error=1-i/4, ssim=i/4)
        self.assertEqual(before, choose(values, list(reversed(values)), 'brisque'))

    def test_exact_subset_expectation_and_budget(self):
        values = self.candidates()
        self.assertAlmostEqual(selection(values, list(values), 1, 'brisque')['error'], .375)
        self.assertEqual(selection(values, list(values), 2, 'brisque')['subsets'], 6)
        self.assertEqual(selection(values, list(values), 4, 'brisque')['error'], .75)
        with self.assertRaises(ValueError):
            selection(values, list(values), 5, 'brisque')

    def test_duplicate_inputs_cannot_create_budget_effect(self):
        values = self.candidates()
        values = {k: values['0'].copy() for k in values}
        for k in (1, 2, 4):
            result = selection(values, list(values), k, 'brisque')
            self.assertEqual(result['error'], 0)
            self.assertEqual(result['brisque'], 4)


if __name__ == '__main__':
    unittest.main()

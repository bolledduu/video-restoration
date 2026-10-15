import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import llbench_pilot as pilot
from llbench_select import select_file


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.fitted = (np.array([.5, .1]), {}, {
            ('task', 'a'): np.array([.9, .3]),
            ('task', 'b'): np.array([.85, .01])})
        self.rows = [dict(task='task', model=m, model_type='large_generative_model') for m in ('a', 'b')]

    def test_metadata_only_choice_and_tradeoff(self):
        self.assertEqual(pilot.choose(self.rows, self.fitted), 'a')
        self.assertEqual(pilot.choose(self.rows, self.fitted, 1), 'b')
        poisoned = [dict(r, risk=999, quality=-999, bt=float('nan')) for r in self.rows]
        self.assertEqual(pilot.choose(poisoned, self.fitted, 1), 'b')

    def test_training_and_development_guards(self):
        row = dict(task='task', model='a', quality=1, risk=0, split='test')
        with self.assertRaises(ValueError):
            pilot.fit([[row]], 10)
        with self.assertRaises(ValueError):
            pilot.select_penalty({'x': [row]}, self.fitted, {})

    def test_group_assignment_is_stable(self):
        self.assertEqual(pilot.split_group('dataset|same.png', 'seed'), pilot.split_group('dataset|same.png', 'seed'))
        self.assertEqual({pilot.split_group(f'dataset|{i}.png', 'seed') for i in range(100)}, {'train', 'dev', 'test'})

    def test_round_trip_without_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pilot.save_selector(self.fitted, 1, directory / 'selector.json')
            (directory / 'input.json').write_text(json.dumps({'example': self.rows}), encoding='utf-8')
            self.assertEqual(select_file(directory / 'input.json', directory / 'selector.json'), {'example': 'b'})
            self.rows[0]['risk'] = 0
            (directory / 'input.json').write_text(json.dumps({'example': self.rows}), encoding='utf-8')
            with self.assertRaises(ValueError):
                select_file(directory / 'input.json', directory / 'selector.json')

    def test_duplicate_and_corrupt_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / 'duplicate.csv'
            path.write_text('trial_id,model\nx,a\nx,a\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                pilot.read_unique(path)
            (directory / 'images.csv').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                pilot.fetch(directory, {'dataset': 'unused', 'revision': 'unused'})

    def test_bootstrap_identity_and_pairing(self):
        rows = [dict(trial_id=str(i), group=str(i), risk=i % 2, quality=.8) for i in range(8)]
        result = pilot.paired_interval(rows, rows, 100, 1)
        self.assertEqual(result['risk_reduction_ci95'], [0, 0])
        self.assertEqual(result['quality_loss_ci95'], [0, 0])
        with self.assertRaises(ValueError):
            pilot.paired_interval(rows, list(reversed(rows)), 100, 1)

    def test_unseen_models_have_deterministic_fallback(self):
        self.assertEqual(pilot.choose(self.rows, (np.array([.5, .1]), {}, {})), 'a')


if __name__ == '__main__':
    unittest.main()

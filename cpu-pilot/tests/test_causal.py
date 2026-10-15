import copy
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_causal import speaker_metrics, safeguards, verified_tracks
from prepare_causal import canonical_sha
from run_causal_omnivr import jobs


class CausalTests(unittest.TestCase):
    def test_provisional_labels_rejected(self):
        with self.assertRaises(ValueError):
            verified_tracks({'status': 'PROVISIONAL'}, 2, 512, 288)

    def test_verified_empty_tracks_rejected(self):
        with self.assertRaises(ValueError):
            verified_tracks(dict(status='VERIFIED', reviewer='test', frames=[dict(index=0, faces=[])]), 1, 512, 288)

    def test_valid_track_and_invalid_box(self):
        annotation = dict(status='VERIFIED', reviewer='fixture reviewer', frames=[dict(index=0, faces=[dict(track_id='face', box_xyxy=[0, 0, 20, 20], speaking=0)])])
        self.assertEqual(verified_tracks(annotation, 1, 512, 288)['face'][1].tolist(), [0])
        bad = copy.deepcopy(annotation)
        bad['frames'][0]['faces'][0]['box_xyxy'][2] = 600
        with self.assertRaises(ValueError):
            verified_tracks(bad, 1, 512, 288)
        bad['frames'][0]['faces'][0]['box_xyxy'] = [0.1, 0, 0.9, 20]
        with self.assertRaises(ValueError):
            verified_tracks(bad, 1, 512, 288)

    def test_unknown_labels_not_scored(self):
        result = speaker_metrics(np.array([0, 1, -1]), np.array([.8, .9, 1.0]))
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['error_rate'], .5)
        self.assertEqual(result['false_positive_rate'], 1.0)
        self.assertAlmostEqual(result['mean_wrong_class_probability'], .45)

    def test_matched_coverage_and_abstention(self):
        result = safeguards(np.array([0, 0, 1, 1]), np.array([.9, .1, .9, .8]), np.array([.1, .1, .9, .8]))
        self.assertEqual(result[0]['coverage'], result[1]['coverage'])
        self.assertEqual(result[0]['error_rate'], 0)
        self.assertIsNone(result[-1]['error_rate'])

    def test_job_pairing_and_repeats(self):
        manifest = {'records': [{'id': 'x'}], 'protocol': {'seeds': [0, 1], 'conditions': ['matching', 'swapped', 'audio_disabled'], 'repeat_matching_seed_zero': True}}
        planned = list(jobs(manifest))
        self.assertEqual(len(planned), 7)
        self.assertEqual(planned[-1][1:], (0, 'matching_repeat'))

    def test_protocol_digest_is_order_independent(self):
        self.assertEqual(canonical_sha({'a': 1, 'b': 2}), canonical_sha({'b': 2, 'a': 1}))


if __name__ == '__main__':
    unittest.main()

import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_gemini_av_pilot import validate_answer,interval_truth,score_answer,paired_counts,make_payload

class GeminiPilotTests(unittest.TestCase):
    def test_labels_are_aggregated_on_the_full_100_frame_grid(self):
        truth,majority=interval_truth(np.array([0]*25+[1]*25+[0]*12+[1]*13+[0]*25))
        self.assertEqual(truth,[0,1,1,0]);self.assertEqual(majority,0)
        with self.assertRaises(ValueError):interval_truth(np.zeros(99))
    def test_incomplete_or_invalid_answers_fail(self):
        for a in ({'decisions':[0]*3,'majority_speaking':0},{'decisions':[True]*4,'majority_speaking':0},{'decisions':[0]*4,'majority_speaking':2}):
            with self.assertRaises(ValueError):validate_answer(a)
    def test_zero_coverage_is_not_perfect_accuracy(self):
        ev=score_answer({'decisions':[-1]*4,'majority_speaking':0},[0]*4,0)
        self.assertEqual(ev['answered'],0);self.assertIsNone(ev['error_rate'])
    def test_harms_and_corrections_on_joint_coverage(self):
        c={'decisions':[0,0,1,-1],'majority_speaking':0}
        r={'decisions':[1,1,-1,0],'majority_speaking':0}
        self.assertEqual(paired_counts(c,r,[0,1,0,0]),dict(jointly_answered=2,captured_errors=1,restored_errors=1,harm=1,corrections=1))
    def test_request_has_no_labels_source_or_candidate_name(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'captured_person_labels.mp4';p.write_bytes(b'fixture')
            data=make_payload(p);serialized=json.dumps(data)
            self.assertNotIn(str(p),serialized);self.assertNotIn('captured_person',serialized)
            self.assertEqual(data['contents'][0]['parts'][0]['videoMetadata']['fps'],24)
if __name__=='__main__':unittest.main()

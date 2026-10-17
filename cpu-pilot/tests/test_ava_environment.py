import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ava_evidence_environment import EvidenceEnvironment, compact_scores, decision_intervals

def fixture():
    return {'secret_ground_truth':[1,0], 'source':'PRIVATE', 'candidates':{
        'captured':{'musiq':20.,'brisque':50.,'probability':[.8,.2],'error':0,'labels':[1,0]},
        'fastdvdnet':{'musiq':25.,'brisque':40.,'probability':[.7,.8],'error':.5}}}

class EnvironmentTests(unittest.TestCase):
    def test_interval_submission_covers_grid_exactly(self):
        env=EvidenceEnvironment(fixture());env.call('inspect_speaking',{'candidate':'captured'})
        args={'candidate':'captured','intervals':[{'start':0,'end':2,'decision':'abstain'}],'rationale':'test'}
        env.call('submit_decisions',args)
        self.assertEqual(env.submission['decisions'],['abstain','abstain'])
    def test_interval_submission_rejects_gap(self):
        env=EvidenceEnvironment(fixture());env.call('inspect_speaking',{'candidate':'captured'})
        with self.assertRaises(ValueError):env.call('submit_decisions',{'candidate':'captured','intervals':[{'start':1,'end':2,'decision':'speaking'}],'rationale':'test'})
    def test_intervals_are_lossless_for_decisions(self):
        scores=[.1,.2,.5,.9,.49]
        expanded=[]
        for start,end,label in decision_intervals(scores):
            self.assertEqual(start,len(expanded));expanded.extend([label]*(end-start))
        self.assertEqual(expanded,[int(p>=.5) for p in scores])
    def test_compact_scores_preserve_threshold(self):
        scores=[0,.4999999999,.5,.50000000001,1]
        self.assertEqual([x>=.5 for x in scores],[x>=.5 for x in compact_scores(scores)])
    def test_no_ground_truth_leaks(self):
        env=EvidenceEnvironment(fixture())
        for key in ('secret_ground_truth','PRIVATE','labels','error'):
            self.assertNotIn(key,json.dumps(env.data))
    def test_restored_selection_requires_comparison(self):
        env=EvidenceEnvironment(fixture())
        for n in ('captured','fastdvdnet'):env.call('inspect_speaking',{'candidate':n})
        args={'candidate':'fastdvdnet','decisions':['speaking','abstain'],'rationale':'uncertain change'}
        with self.assertRaises(ValueError):env.call('submit_decisions',args)
        self.assertIsNone(env.submission)
        self.assertEqual(env.call('compare_with_capture',{'candidate':'fastdvdnet'})['disagreement'],.5)
        self.assertEqual(env.call('submit_decisions',args)['answered'],1)
        with self.assertRaises(ValueError):env.call('list_evidence',{})
    def test_invalid_calls_consume_budget(self):
        env=EvidenceEnvironment(fixture(),budget=1)
        with self.assertRaises(ValueError):env.call('inspect_speaking',{'candidate':'reference'})
        with self.assertRaises(ValueError):env.call('list_evidence',{})
    def test_malformed_arguments_consume_budget(self):
        env=EvidenceEnvironment(fixture(),budget=1)
        with self.assertRaises(ValueError):env.call('list_evidence',[])
        self.assertEqual(len(env.trace),1)
        with self.assertRaises(ValueError):env.call('list_evidence',{})
    def test_grid_mismatch_rejected(self):
        data=fixture();data['candidates']['fastdvdnet']['probability']=[.5]
        with self.assertRaises(ValueError):EvidenceEnvironment(data)
    def test_payload_is_detached_from_evaluator(self):
        data=fixture();env=EvidenceEnvironment(data)
        data['candidates']['captured']['probability'][0]=0
        self.assertEqual(env.call('inspect_interval',{'candidate':'captured','start':0,'end':1})['probability'][0],.8)
    def test_verify_captured_temporal_support_tool_execution(self):
        data = fixture()
        data['candidates']['fastdvdnet']['cts'] = {'cts_support': 12.5, 'mouth_specificity': 8.0, 'heldout_mse': 45.0, 'mask_stability': 10.0, 'diagnostic_validity': 'VALID'}
        env = EvidenceEnvironment(data)
        out = env.call('verify_captured_temporal_support', {'candidate': 'fastdvdnet'})
        self.assertEqual(out['candidate'], 'fastdvdnet')
        self.assertEqual(out['cts_support'], 12.5)
        self.assertEqual(out['recommendation'], 'SAFE_TO_RESTORE')
        self.assertEqual(out['provenance'], 'withheld_pixel_temporal_audit')
        self.assertNotIn('labels', out)
        self.assertNotIn('secret_ground_truth', out)

if __name__=='__main__':unittest.main()

import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from captured_support import masked_fill,fixed_mask,region_mask,probes,support_scores,transition_targets

class CapturedSupportTests(unittest.TestCase):
    def test_withheld_values_do_not_affect_fill_or_audit_prediction(self):
        rng=np.random.default_rng(8);video=rng.uniform(0,255,(15,24,24,3));mask=fixed_mask(24,24,1729)
        changed=video.copy();changed[:,mask]=rng.uniform(-1e6,1e6,changed[:,mask].shape)
        a=masked_fill(video,mask);b=masked_fill(changed,mask)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[:,~mask],video[:,~mask])
    def test_no_temporal_wrap_or_off_region_changes(self):
        b=np.zeros((15,12,12,3));r=np.broadcast_to(np.arange(15)[:,None,None,None],b.shape).copy();region=region_mask(12,12,'lower_face')
        valid,variants=probes(b,r,region)
        np.testing.assert_array_equal(valid,np.arange(3,12))
        for name,v in variants.items():
            np.testing.assert_array_equal(v[:,~region],r[:,~region])
            if name.startswith('shift'):
                np.testing.assert_array_equal(v[:3],r[:3]);np.testing.assert_array_equal(v[-3:],r[-3:])
        np.testing.assert_array_equal(variants['shift_3'][3,region],r[6,region])
    def test_static_increment_has_zero_timing_contrast(self):
        rng=np.random.default_rng(8);y=rng.uniform(0,255,(15,24,24,3));mask=fixed_mask(24,24,1729);b=masked_fill(y,mask)
        scores,_=support_scores(y,b,b+2,mask,region_mask(24,24,'lower_face'))
        np.testing.assert_allclose(scores['temporal_support'],0,atol=1e-14)
    def test_all_four_transition_classes(self):
        np.testing.assert_array_equal(transition_targets([1]*4,[.8,.8,.2,.2],[.8,.2,.8,.2]),[0,1,2,3])
    def test_fixed_masks_and_rgb_exclusion(self):
        np.testing.assert_array_equal(fixed_mask(20,20,1),fixed_mask(20,20,1))
        self.assertEqual(int(fixed_mask(20,20,1).sum()),40)
class SupportToolTests(unittest.TestCase):
    def test_tool_rejects_outcomes_and_returns_detached_copy(self):
        from captured_support_tool import SupportEvidenceEnvironment
        measured={'candidates':{'captured':{'musiq':20,'brisque':40,'probability':[.1,.8]},'fastdvdnet':{'musiq':30,'brisque':30,'probability':[.2,.9]}}}
        assessment=dict(estimated_harm=.1,estimated_correction=.2,lower_face_support=.01,audit_instability=.02,recommended_candidate='fastdvdnet',assessor_model_sha256='fixture',scope='test fixture')
        with self.assertRaises(ValueError):SupportEvidenceEnvironment(measured,{**assessment,'labels':[0,1]})
        env=SupportEvidenceEnvironment(measured,assessment)
        value=env.call('assess_restoration',{'candidate':'fastdvdnet'});value['estimated_harm']=999
        self.assertEqual(env.call('assess_restoration',{'candidate':'fastdvdnet'})['estimated_harm'],.1)
        self.assertEqual(len(env.trace),2)
    def test_invalid_risk_rejected(self):
        from captured_support_tool import SupportEvidenceEnvironment
        measured={'candidates':{'captured':{'musiq':20,'brisque':40,'probability':[.1]},'fastdvdnet':{'musiq':30,'brisque':30,'probability':[.2]}}}
        a=dict(estimated_harm=-1,estimated_correction=.2,lower_face_support=.01,audit_instability=.02,recommended_candidate='captured',assessor_model_sha256='fixture',scope='test')
        with self.assertRaises(ValueError):SupportEvidenceEnvironment(measured,a)
class SupportFeatureTests(unittest.TestCase):
    def fixture(self):
        import copy
        indices=list(range(9));region=dict(indices=[3,4,5],support=[.01,.02,.03],mse={'factual':[.2,.2,.2]},probe_probabilities={name:[.4]*9 for name in ('shift_-3','shift_3','static','scale_0.5','scale_1.5')})
        fold=dict(audit_probability=[.3]*9,regions={name:copy.deepcopy(region) for name in ('lower_face','cheek','full_frame')})
        return dict(id='fixture',source='fixture',indices=indices,labels=[0]*9,transition_targets=[0]*9,captured_probability=[.2]*9,restored_probability=[.3]*9,quality={'captured':{'musiq':20,'brisque':40},'fastdvdnet':{'musiq':30,'brisque':30}},candidate_key='fastdvdnet',folds=[copy.deepcopy(fold),copy.deepcopy(fold)])
    def test_feature_extraction_does_not_read_labels(self):
        from evaluate_captured_support import extract
        c=self.fixture();before,pos=extract(c);c['labels']=[1]*9;c['transition_targets']=[3]*9
        after,pos2=extract(c);np.testing.assert_array_equal(before,after);np.testing.assert_array_equal(pos,[3,4,5])
    def test_missing_transition_classes_block_training(self):
        from evaluate_captured_support import fit
        with self.assertRaisesRegex(ValueError,'lack'):fit([self.fixture()],'support')
if __name__=='__main__':unittest.main()

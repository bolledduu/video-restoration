import copy, sys, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_full_benchmark import build_cases
from run_evidence_agent import Provider, score
from run_final_experiment import evaluate_policy_on_pair

def pair(identifier='source__person__silent__sigma25__fastdvdnet', restorer='fastdvdnet'):
    return dict(id=identifier,restorer=restorer,source='source',dataset='fixture',degradation='sigma25',
        labels=np.array([0,1]),captured_prob=np.array([.1,.8]),restored_prob=np.array([.2,.9]),
        captured_musiq=20,restored_musiq=30,captured_brisque=40,restored_brisque=30)

class AuditRegressionTests(unittest.TestCase):
    def test_person_state_and_degradation_remain_distinct(self):
        a=pair();b=pair(a['id'].replace('silent','speaking'));c=pair(a['id'].replace('sigma25','sigma40'))
        self.assertEqual(len(build_cases([a,b,c])),3)
    def test_alternatives_merge_without_losing_identity(self):
        a=pair();b=pair(a['id'].replace('fastdvdnet','spatial_nlm'),'spatial_nlm')
        self.assertEqual(len(build_cases([a,b])),1)
        b['labels']=np.array([1,0])
        with self.assertRaises(ValueError):build_cases([a,b])
    def test_conflicting_capture_rejected(self):
        a=pair();b=pair(a['id'].replace('fastdvdnet','spatial_nlm'),'spatial_nlm')
        b['captured_prob']=np.array([.3,.8])
        with self.assertRaises(ValueError):build_cases([a,b])
    def test_same_predictions_do_not_mean_same_evidence(self):
        p=pair();p['restored_prob']=np.array([.2,.7])
        row=evaluate_policy_on_pair('confidence_max',p)
        self.assertEqual(row['restored_selected_frames'],0)
        self.assertIsNone(row['delta_musiq'])
    def test_empty_coverage_has_undefined_error(self):
        p=pair();p['restored_prob']=1-p['captured_prob']
        row=evaluate_policy_on_pair('disagreement_abstain',p)
        self.assertEqual(row['answered_frames'],0);self.assertIsNone(row['error_rate'])
    def test_short_submission_rejected(self):
        case={'labels':[0,1],'candidates':{'captured':{'probability':[.1,.9]}}}
        with self.assertRaises(ValueError):score(case,{'submission':{'decisions':['speaking']}})
    def test_legacy_completed_status_and_abstention_are_rescored(self):
        from agent_result_adapters import legacy_runs
        c={'id':'a','label':0,'candidates':{'captured':{'probability':[.1,.9]}}}
        run={'case_id':'a','status':'completed','condition':'captured','repeat':0,
             'submission':{'frame_tags':[-1,0]},'evaluation':{'frame_wrong':0}}
        document={'cases':[c],'protocol':{'model':'fixture'},'runs':[run]}
        row=legacy_runs(document,'fixture')[0]
        self.assertEqual(row['evaluation']['answered'],1)
        self.assertEqual(row['evaluation']['captured_errors_on_same_answered_frames'],1)
        del run['evaluation']['frame_wrong']
        with self.assertRaises(KeyError):legacy_runs(document,'fixture')
    def test_audio_resampling_rounding_does_not_pad_missing_audio(self):
        from run_ava_local import trim_audio_rounding
        audio=np.arange(64004)
        normalized,excess=trim_audio_rounding(audio,64000)
        self.assertEqual(excess,4);self.assertEqual(len(normalized),64000)
        with self.assertRaises(RuntimeError):trim_audio_rounding(audio[:63999],64000)
        with self.assertRaises(RuntimeError):trim_audio_rounding(np.zeros(64100),64000)
    def test_backup_authorization_is_rebuilt(self):
        class Response:
            headers={}
            def __init__(self,code):self.status_code=code;self.ok=code==200
            def json(self):return {'ok':True}
        class Session:
            def __init__(self):self.headers=[]
            def request(self,*a,**kw):
                self.headers.append(kw['headers']['Authorization']);return Response(401 if len(self.headers)==1 else 200)
        with patch('run_evidence_agent.credential',side_effect=['fixture-primary','fixture-backup']):p=Provider('groq')
        p.session=Session();p.request('GET','/models')
        self.assertEqual(p.session.headers,['Bearer fixture-primary','Bearer fixture-backup'])
if __name__=='__main__':unittest.main()

import sys, unittest, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_evidence_agent import blind_case, score, run_episode

def case():
    return {'labels':[0,1,1], 'source':'hidden', 'candidates':{
        'captured':{'probability':[.1,.9,.1],'musiq':20,'brisque':40},
        'fastdvdnet':{'probability':[.9,.9,.9],'musiq':30,'brisque':30},
        'reference':{'probability':[0,1,1],'musiq':60,'brisque':10}}}

class RunnerTests(unittest.TestCase):
    def test_controller_forces_finalization_after_inspection(self):
        class InspectingProvider:
            def __init__(self):self.turn=0
            def request(self,*args,**kwargs):
                payload=kwargs['json'];self.turn+=1
                if isinstance(payload['tool_choice'],dict):
                    calls=[('submit_decisions',{'candidate':'captured','decisions':['not_speaking','speaking','not_speaking'],'rationale':'test fixture'})]
                else:
                    calls=[('inspect_speaking',{'candidate':c}) for c in ('captured','fastdvdnet')]+[('compare_with_capture',{'candidate':'fastdvdnet'})]
                return {'model':'fixture','choices':[{'message':{'role':'assistant','content':None,'tool_calls':[
                    {'id':str(i),'type':'function','function':{'name':n,'arguments':json.dumps(a)}} for i,(n,a) in enumerate(calls)]}}]}
        provider=InspectingProvider();result=run_episode(provider,'fixture',case(),'groq')
        self.assertEqual(provider.turn,5);self.assertEqual(result['status'],'COMPLETE')
    def test_reference_excluded(self):
        self.assertNotIn('reference',blind_case(case())['candidates'])
        self.assertEqual(set(blind_case(case())),{'candidates'})
    def test_paired_harm_and_correction_not_net_only(self):
        result=score(case(),{})['restorers']['fastdvdnet']
        self.assertEqual(result,{'errors':1,'harm':1,'corrections':1})
    def test_same_coverage_comparison(self):
        result=score(case(),{'submission':{'decisions':['abstain','speaking','speaking']}})
        self.assertEqual(result['agent_errors'],0)
        self.assertEqual(result['captured_errors_on_same_answered_frames'],1)
        self.assertEqual(result['answered'],2)
    def test_agent_harm_and_correction_use_answered_indices(self):
        result=score(case(),{'submission':{'decisions':['speaking','abstain','speaking']}})
        self.assertEqual(result['agent_harm'],1)
        self.assertEqual(result['agent_corrections'],1)
        self.assertEqual(result['agent_errors'],1)
        result=score(case(),{'submission':{'decisions':['abstain']*3}})
        self.assertIsNone(result['agent_error_rate'])
        self.assertEqual(result['answered'],0)
    def test_failure_preserves_status_not_fake_answer(self):
        class FailedProvider:
            def request(self,*args,**kwargs):raise RuntimeError('Provider HTTP 429')
        result=run_episode(FailedProvider(),'test',case(),'groq')
        self.assertEqual(result['status'],'PROVIDER_FAILED');self.assertIsNone(result['submission'])
        self.assertNotIn('hidden',str(result['messages']))

if __name__=='__main__':unittest.main()

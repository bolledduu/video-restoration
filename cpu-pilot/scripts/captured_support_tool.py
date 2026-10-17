"""Typed, label-blind interface for an independently fitted support selector."""
import copy
import numpy as np
from ava_evidence_environment import EvidenceEnvironment,schemas
from evaluate_captured_support import extract,BASE,AUDIT,SUPPORT

FIELDS={'estimated_harm','estimated_correction','lower_face_support','audit_instability','recommended_candidate','assessor_model_sha256','scope'}

def assess(case,model,tau,model_sha256):
    """The model must be fitted/calibrated outside inference; no labels read here."""
    x,pos=extract(case);coef=np.asarray(model['coef']);mean=np.asarray(model['mean']);scale=np.asarray(model['scale'])
    if model['classes']!=[0,1,2,3] or coef.shape!=(4,len(mean)):raise ValueError('Invalid four-state artifact')
    if not np.isfinite(coef).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Invalid model parameters')
    z=((x[:,:len(mean)]-mean)/scale)@coef.T+np.asarray(model['intercept'])
    z-=z.max(axis=1,keepdims=True);probs=np.exp(z);probs/=probs.sum(axis=1,keepdims=True)
    harm=float(probs[:,1].mean());correction=float(probs[:,2].mean())
    quality=case['quality'][case['candidate_key']]['musiq']>case['quality']['captured']['musiq']
    return dict(estimated_harm=harm,estimated_correction=correction,
        lower_face_support=float(x[:,len(BASE)+len(AUDIT)].mean()),audit_instability=float(x[:,len(BASE)+1].mean()),
        recommended_candidate='fastdvdnet' if quality and harm<=tau and correction>=harm else 'captured',
        assessor_model_sha256=model_sha256,
        scope='Estimated TalkNet transition risk; not calibrated Gemini error probability or a visual-truth certificate')

class SupportEvidenceEnvironment(EvidenceEnvironment):
    def __init__(self,measured,assessment,budget=24):
        super().__init__(measured,budget)
        if set(assessment)!=FIELDS:raise ValueError('Unexpected assessor fields; reject label or outcome leakage')
        for key in ('estimated_harm','estimated_correction','lower_face_support','audit_instability'):
            if type(assessment[key]) not in (int,float) or not np.isfinite(assessment[key]):raise ValueError('Invalid assessor number')
        if not 0<=assessment['estimated_harm']<=1 or not 0<=assessment['estimated_correction']<=1:raise ValueError('Invalid risk estimate')
        if assessment['estimated_harm']+assessment['estimated_correction']>1+1e-8:raise ValueError('Mutually exclusive transition probabilities exceed one')
        if assessment['audit_instability']<0:raise ValueError('Negative instability')
        if assessment['recommended_candidate'] not in self.data:raise ValueError('Unavailable recommendation')
        self.assessment=copy.deepcopy(assessment)
    def _dispatch(self,name,args):
        if name=='assess_restoration':
            if args!={'candidate':'fastdvdnet'}:raise ValueError('Assessor supports only the measured FastDVDnet candidate')
            return copy.deepcopy(self.assessment)
        return super()._dispatch(name,args)

def support_schemas():
    return schemas()+[{'type':'function','function':{'name':'assess_restoration','description':'Inspect captured-support diagnostics and estimated detector-transition risk; not ground truth.',
        'parameters':{'type':'object','properties':{'candidate':{'type':'string','enum':['fastdvdnet']}},'required':['candidate'],'additionalProperties':False}}}]

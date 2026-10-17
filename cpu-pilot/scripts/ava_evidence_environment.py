"""Label-blind bounded evidence tools for the first AVA agent experiment.

Consumes measured candidates, never labels or source identifiers. Test fixtures
are software checks, not model observations. This module makes no API requests.
"""
import copy
import hashlib
import json
import math

def compact_scores(values):
    # Truncate display precision; keep the >=0.5 decision identical, including ties.
    return [math.floor(p*1000000)/1000000 for p in values]

def decision_intervals(values):
    intervals=[]
    for i,p in enumerate(values):
        decision=int(p>=.5)
        if intervals and intervals[-1][2]==decision:intervals[-1][1]=i+1
        else:intervals.append([i,i+1,decision])
    return intervals

class EvidenceEnvironment:
    def __init__(self, measured, budget=24):
        self.data = {}
        for name, value in measured['candidates'].items():
            if name not in {'captured', 'fastdvdnet', 'spatial_nlm'}:
                raise ValueError('Unsupported candidate')
            # Explicit allowlist prevents label, error, reference, and source leakage.
            self.data[name] = {k:copy.deepcopy(value[k]) for k in ('musiq','brisque','probability')}
            if not self.data[name]['probability']:
                raise ValueError('Empty perception output')
            for p in self.data[name]['probability']:
                if type(p) not in (int,float) or not math.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError('Invalid detector score')
            for key in ('musiq','brisque'):
                if type(value[key]) not in (int,float) or not math.isfinite(value[key]):
                    raise ValueError('Invalid quality score')
        if 'captured' not in self.data: raise ValueError('Captured input is required')
        self.n=len(self.data['captured']['probability'])
        if any(len(v['probability'])!=self.n for v in self.data.values()):
            raise ValueError('Scoring grids differ')
        self.budget=budget; self.trace=[]; self.inspected=set(); self.compared=set(); self.submission=None
        self.evidence_hash=hashlib.sha256(json.dumps(self.data,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def call(self, name, args):
        if self.submission is not None: raise ValueError('Episode already finalized')
        if len(self.trace)>=self.budget: raise ValueError('Tool budget exhausted')
        event={'tool':name,'arguments':copy.deepcopy(args)}
        self.trace.append(event)  # Invalid calls consume budget too.
        try:
            if not isinstance(args,dict): raise ValueError('Arguments must be an object')
            output=self._dispatch(name,args)
            event['output']=copy.deepcopy(output)
            return output
        except (ValueError,KeyError,TypeError) as error:
            event['error']=str(error)
            raise

    def _dispatch(self,name,args):
        if name=='list_evidence':
            if args: raise ValueError('Unexpected arguments')
            return {'candidates':list(self.data),'scored_timestamps':self.n,'evidence_hash':self.evidence_hash,'access':'tool_measurements_only','scores_are_uncalibrated':True}
        candidate=args.get('candidate')
        if candidate not in self.data: raise ValueError('Unknown candidate')
        value=self.data[candidate]; base=self.data['captured']
        if name=='inspect_quality':
            if set(args)!={'candidate'}: raise ValueError('Unexpected arguments')
            return {'musiq':value['musiq'],'brisque':value['brisque'],'musiq_direction':'higher','brisque_direction':'lower'}
        if name=='inspect_speaking':
            if set(args)!={'candidate'}: raise ValueError('Unexpected arguments')
            self.inspected.add(candidate)
            p=value['probability']
            return {'intervals_start_endExclusive_speaking':decision_intervals(p),'threshold':.5,'calibrated':False,
                    'score_min':min(compact_scores(p)),'score_max':max(compact_scores(p)),
                    'score_mean':round(sum(p)/len(p),6),'detail':'Intervals preserve every threshold decision. inspect_interval provides local scores.'}
        if name=='inspect_interval':
            if set(args)!={'candidate','start','end'}: raise ValueError('Unexpected arguments')
            a,b=args['start'],args['end']
            if type(a) is not int or type(b) is not int or not 0<=a<b<=self.n: raise ValueError('Invalid interval')
            if b-a>20:raise ValueError('Inspect at most 20 timestamps per interval')
            return {'start':a,'end_exclusive':b,'probability':compact_scores(value['probability'][a:b])}
        if name=='compare_with_capture':
            if set(args)!={'candidate'}: raise ValueError('Unexpected arguments')
            self.compared.add(candidate)
            return {'disagreement':sum((a>=.5)!=(b>=.5) for a,b in zip(value['probability'],base['probability']))/self.n,
                    'mean_absolute_score_change':sum(abs(a-b) for a,b in zip(value['probability'],base['probability']))/self.n,
                    'delta_musiq':value['musiq']-base['musiq'],'delta_brisque':value['brisque']-base['brisque']}
        if name=='submit_decisions':
            if set(args) not in ({'candidate','decisions','rationale'},{'candidate','intervals','rationale'}): raise ValueError('Unexpected arguments')
            if 'captured' not in self.inspected or candidate not in self.inspected: raise ValueError('Inspect captured and selected speaking evidence first')
            if candidate!='captured' and candidate not in self.compared: raise ValueError('Compare restored evidence first')
            if 'intervals' in args:
                if isinstance(args['intervals'], str):
                    try: args['intervals'] = json.loads(args['intervals'])
                    except Exception: pass
                labels=[]
                if not isinstance(args['intervals'],list):raise ValueError('Intervals must be an array')
                for interval in args['intervals']:
                    if not isinstance(interval,dict) or set(interval)!={'start','end','decision'}:raise ValueError('Invalid interval fields')
                    a,b,decision=interval['start'],interval['end'],interval['decision']
                    if type(a) is not int or type(b) is not int or a!=len(labels) or not a<b<=self.n:
                        raise ValueError(f"Intervals must partition the complete grid without gaps or overlap from 0 to {self.n} (received interval [{a}, {b}) but expected start={len(labels)} and max end={self.n})")
                    if decision not in ('speaking','not_speaking','abstain'):raise ValueError('Invalid interval decision')
                    labels.extend([decision]*(b-a))
            else:labels=args['decisions']
            if not isinstance(labels,list) or len(labels)!=self.n or any(x not in ('speaking','not_speaking','abstain') for x in labels): raise ValueError('One valid decision per scored timestamp is required')
            if not isinstance(args['rationale'],str) or not args['rationale'].strip(): raise ValueError('Rationale required')
            self.submission=copy.deepcopy(args)
            self.submission['decisions']=copy.deepcopy(labels)
            return {'accepted':True,'answered':sum(x!='abstain' for x in labels),'eligible':self.n}
        raise ValueError('Unknown tool')

def schemas():
    candidate={'type':'string','enum':['captured','fastdvdnet','spatial_nlm']}
    definitions=[('list_evidence','List available evidence and the scored time grid.',{}),
                 ('inspect_quality','Inspect measured appearance quality; not correctness.',{'candidate':candidate}),
                 ('inspect_speaking','Inspect lossless threshold-decision intervals and uncalibrated score summaries.',{'candidate':candidate}),
                 ('inspect_interval','Inspect local scores for at most 20 consecutive scored timestamps.',{'candidate':candidate,'start':{'type':'integer'},'end':{'type':'integer'}}),
                 ('compare_with_capture','Compare restoration with captured evidence; disagreement does not reveal truth.',{'candidate':candidate}),
                 ('submit_decisions','Select evidence and partition ALL timestamps into consecutive decision intervals from 0 to the total count, end exclusive.',{'candidate':candidate,'intervals':{'type':'array','items':{'type':'object','properties':{'start':{'type':'integer'},'end':{'type':'integer'},'decision':{'type':'string','enum':['speaking','not_speaking','abstain']}},'required':['start','end','decision'],'additionalProperties':False}},'rationale':{'type':'string'}})]
    return [{'type':'function','function':{'name':n,'description':d,'parameters':{'type':'object','properties':p,'required':list(p),'additionalProperties':False}}} for n,d,p in definitions]

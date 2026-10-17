"""Provider-neutral, bounded tool agent. Labels are used only after inference."""
import argparse, hashlib, json, os, time
from pathlib import Path
import requests
from ava_evidence_environment import EvidenceEnvironment, schemas

PROVIDERS = {'groq': ('https://api.groq.com/openai/v1', 'GROQ_API_KEY'),
             'gemini': ('https://generativelanguage.googleapis.com/v1beta/openai', 'GEMINI_API_KEY'),
             'openrouter': ('https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY')}
SYSTEM = """You select audiovisual evidence for a visible person's speaking-state decisions.
You have NO direct access to pixels, audio, transcripts, identities or true labels.
Use only the measurement tools. Higher MUSIQ and lower BRISQUE indicate appearance
quality, not attribution correctness. TalkNet probabilities are uncalibrated.
Inspect captured and restored candidates before choosing. Compare restored evidence
with capture before selecting it when comparison tools are available. Disagreement alone does not establish an error.
Speaking inspections return [start, end_exclusive, speaking_0_or_1] intervals,
which preserve every threshold decision. Request local scores only if needed.
After inspecting the candidates and comparisons, submit without repeating inspections.
Submit consecutive intervals with start, exclusive end, and a speaking/not_speaking/abstain
decision using submit_decisions. Cover every timestamp from 0 to the total count.
Prefer useful quality improvements only when evidence supports
them. Do not invent observations. Finish within 8 model turns and 24 tool calls."""

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def credential(name):
    value=os.environ.get(name)
    if not value and os.name=='nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as reg:value=winreg.QueryValueEx(reg,name)[0]
        except OSError: pass
    return value

class Provider:
    def __init__(self, name):
        self.name = name
        self.base,self.variable=PROVIDERS[name];self.key=credential(self.variable)
        self.backup=credential(self.variable+'_BACKUP');self.used_backup=False
        if not self.key:raise RuntimeError('Missing credential: '+self.variable)
        self.session=requests.Session()

    def request(self, method, path, **kwargs):
        for attempt in range(10):
            headers = {'Authorization': 'Bearer ' + self.key}
            if self.name == 'openrouter':
                headers['X-Title'] = 'Video Restoration Benchmark'
            try: r=self.session.request(method,self.base+path,headers=headers,timeout=90,**kwargs)
            except requests.RequestException:raise RuntimeError('Provider transport failed; no response recorded') from None
            if r.status_code==401 and self.backup and not self.used_backup:
                self.key=self.backup;self.used_backup=True;continue
            if r.status_code in (429,500,502,503,504) and attempt<9:
                try:delay=float(r.headers.get('retry-after','6'))
                except ValueError:delay=6
                time.sleep(min(45,max(4,delay)+2));continue
            if not r.ok:raise RuntimeError('Provider HTTP '+str(r.status_code)+'; response body omitted to protect credentials')
            return r.json()
        raise RuntimeError('Provider retries exhausted')


def blind_case(case):
    return {'candidates':{k:v for k,v in case['candidates'].items() if k in ('captured','fastdvdnet','spatial_nlm')}}

def run_episode(provider, model, case, provider_name, assessment=None, allowed_tools=None, max_turns=8):
    if allowed_tools and 'verify_captured_temporal_support' in allowed_tools:
        raise ValueError('Legacy proxy verifier disabled. Supply a measured assessor receipt and use assess_restoration.')
    if type(max_turns) is not int or not 1 <= max_turns <= 12:raise ValueError('Invalid turn budget')
    if assessment is None:
        env=EvidenceEnvironment(blind_case(case));tool_definitions=schemas
    else:
        from captured_support_tool import SupportEvidenceEnvironment, support_schemas
        env=SupportEvidenceEnvironment(blind_case(case),assessment);tool_definitions=support_schemas
    available_definitions=[v for v in tool_definitions() if v['function']['name']!='verify_captured_temporal_support']
    if allowed_tools is not None:
        if not set(allowed_tools)<={v['function']['name'] for v in available_definitions}:raise ValueError('Unknown or unavailable tool')
        available_definitions=[v for v in available_definitions if v['function']['name'] in allowed_tools]
    env.allowed_tools={v['function']['name'] for v in available_definitions}
    has_compare='compare_with_capture' in env.allowed_tools
    if assessment is not None and 'assess_restoration' not in env.allowed_tools:raise ValueError('Assessor provided but its tool is unavailable')
    usage=[];served=[];messages=[{'role':'system','content':SYSTEM.replace('within 8 model turns',f'within {max_turns} model turns')},
        {'role':'user','content':f'The total number of scored timestamps for this clip is exactly {env.n}. Inspect the available candidates, compare restored candidates with capture, select the best candidate, and submit decisions covering all timestamps from 0 to {env.n} using submit_decisions.'}]

    if assessment is not None:
        messages.append({'role':'user','content':'Inspect assess_restoration once before selecting evidence. Its estimates concern detector transitions, not verified truth or calibrated agent-error probabilities.'})
    for turn in range(max_turns):
        ready = (set(env.data) <= env.inspected and
                 ((set(env.data) - {'captured'}) <= env.compared if has_compare else True) and
                 (assessment is None or any(e['tool']=='assess_restoration' and 'output' in e for e in env.trace)))
        tools = available_definitions
        tool_choice = 'auto'
        if ready and turn >= 4:
            tools = [s for s in tools if s['function']['name'] == 'submit_decisions']
            tool_choice = {'type':'function', 'function':{'name':'submit_decisions'}}
        elif turn >= 5:
            messages.append({'role':'user', 'content':
                'Complete missing candidate inspections and comparisons, then submit. Do not repeat completed inspections.'})
        payload={'model':model,'messages':messages,'tools':tools,'tool_choice':tool_choice,'temperature':0,'max_completion_tokens':1024}


        if provider_name == 'gemini': payload['reasoning_effort'] = 'low'
        try:
            data = provider.request('POST', '/chat/completions', json=payload)
        except RuntimeError as error:
            return {'status': 'PROVIDER_FAILED', 'submission': None, 'error': str(error), 'evidence_sha256': env.evidence_hash,
                    'served_models': served, 'usage': usage, 'messages': messages, 'tool_trace': env.trace}
        usage.append(data.get('usage',{}));served.append(data.get('model'))
        msg=data['choices'][0]['message'];messages.append(msg)
        calls=msg.get('tool_calls') or []
        for call in calls:
            try:output=env.call(call['function']['name'],json.loads(call['function']['arguments']))
            except (ValueError,KeyError,TypeError) as error:output={'error':str(error)}
            messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(output,allow_nan=False)})
        if env.submission:break
        if len(env.trace)>=env.budget:break
        if not calls:messages.append({'role':'user','content':'Submit using submit_decisions; plain text does not complete the episode.'})
        time.sleep(1)
    return {'status':'COMPLETE' if env.submission else 'BUDGET_EXHAUSTED','submission':env.submission,
            'evidence_sha256':env.evidence_hash,'served_models':served,'usage':usage,'messages':messages,'tool_trace':env.trace}

def score(case, run):
    """Paired counts, not independent-frame significance tests."""
    truth=case['labels'];base=[int(p>=.5) for p in case['candidates']['captured']['probability']]
    if not truth or len(truth)!=len(base):raise ValueError('Label grid mismatch')
    if any(type(y) not in (int, bool) or y not in (0,1) for y in truth):raise ValueError('Invalid labels')
    result={'eligible':len(truth),'captured_errors':sum(a!=b for a,b in zip(base,truth)),'restorers':{}}
    for name,value in case['candidates'].items():
        if name in ('captured','reference'):continue
        pred=[int(p>=.5) for p in value['probability']]
        if len(pred)!=len(truth):raise ValueError('Candidate grid mismatch')
        result['restorers'][name]={'errors':sum(p!=y for p,y in zip(pred,truth)),
            'harm':sum(b==y and p!=y for b,p,y in zip(base,pred,truth)),
            'corrections':sum(b!=y and p==y for b,p,y in zip(base,pred,truth))}
    if run.get('submission'):
        decisions=run['submission']['decisions']
        if len(decisions)!=len(truth):raise ValueError('Incomplete decision grid')
        if any(v not in ('speaking','not_speaking','abstain') for v in decisions):raise ValueError('Invalid decision')
        pairs=[(i,int(v=='speaking')) for i,v in enumerate(decisions) if v!='abstain']
        result.update(answered=len(pairs),coverage=len(pairs)/len(truth),
            agent_errors=sum(p!=truth[i] for i,p in pairs),
            captured_errors_on_same_answered_frames=sum(base[i]!=truth[i] for i,p in pairs),
            agent_error_rate=sum(p!=truth[i] for i,p in pairs)/len(pairs) if pairs else None,
            agent_harm=sum(base[i]==truth[i] and p!=truth[i] for i,p in pairs),
            agent_corrections=sum(base[i]!=truth[i] and p==truth[i] for i,p in pairs))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--measurements',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--assessments',type=Path);ap.add_argument('--provider',choices=PROVIDERS,required=True);ap.add_argument('--model',required=True);args=ap.parse_args()
    measured=json.loads(args.measurements.read_text())
    assessments=None
    if args.assessments:
        receipt=json.loads(args.assessments.read_text())
        if receipt['measurements_sha256']!=sha(args.measurements):raise ValueError('Assessor/measurement identity mismatch')
        artifact=Path(receipt['model_artifact'])
        if not artifact.exists():artifact=args.assessments.parent/artifact.name
        if sha(artifact)!=receipt['model_artifact_sha256']:raise ValueError('Frozen assessor artifact changed')
        if json.loads(artifact.read_text())['status']!='FROZEN_BEFORE_TEST':raise ValueError('Assessor is not frozen')
        assessments=receipt['assessments']
        if set(assessments)!={c['id'] for c in measured['cases']}:raise ValueError('Assessor case set mismatch')
    if measured['status'] not in ('COMPLETE','COMPLETE_WITH_EXCLUSIONS'):raise RuntimeError('Measurements are incomplete')
    if not measured['cases']:raise RuntimeError('No eligible cases')
    identity={'measurements_sha256':sha(args.measurements),'runner_sha256':sha(__file__),
        'assessments_sha256':sha(args.assessments) if args.assessments else None,
        'support_tool_sha256':sha(Path(__file__).with_name('captured_support_tool.py')) if args.assessments else None,
        'environment_sha256':sha(Path(__file__).with_name('ava_evidence_environment.py')),'provider':args.provider,'model':args.model,
        'max_turns':8,'max_tools':24,'max_completion_tokens':1024,'access':'tool_measurements_only','temperature':0,
        'finalization_policy':'Force submission from turn 5 only after all inspections and comparisons; incomplete evidence exhausts budget','submission_format':'complete interval partition'}

    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'identity':identity,'status':'RUNNING','cases':[]}
    if args.output.exists():
        report=json.loads(args.output.read_text())
        if report['identity']!=identity:raise RuntimeError('Resume identity mismatch; use a new output path')
    provider=Provider(args.provider)
    available=provider.request('GET','/models')
    if args.model not in {x['id'] for x in available['data']}:raise RuntimeError('Exact requested model unavailable; no substitution')
    done={c['id'] for c in report['cases'] if c['run']['status']=='COMPLETE'}
    if any(c['run']['status']!='COMPLETE' for c in report['cases']):raise RuntimeError('Incomplete episode exists; preserve it and use a new output path for a retry')
    for case in measured['cases']:
        if case['id'] in done:continue
        save(args.output,report)
        try:run=run_episode(provider,args.model,case,args.provider,assessments[case['id']] if assessments else None)
        except Exception as error:
            report['status']='FAILED';report['failure']={'case':case['id'],'type':type(error).__name__,'message':str(error)};save(args.output,report);raise
        report['cases'].append({'id':case['id'],'run':run,'evaluation':score(case,run)})
        save(args.output,report);print(case['id'],run['status'],flush=True)
        if run['status']!='COMPLETE':break
    report['status']='COMPLETE' if len(report['cases'])==len(measured['cases']) and all(c['run']['status']=='COMPLETE' for c in report['cases']) else 'INCOMPLETE'
    report.pop('failure',None);save(args.output,report)
    if report['status']!='COMPLETE':raise SystemExit(2)

if __name__=='__main__':main()

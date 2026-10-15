"""Bounded Qwen tool agent. Credentials never enter artifacts; labels never enter prompts."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import requests
ROOT=Path(__file__).resolve().parents[1]
BASE='https://api.groq.com/openai/v1'
NAMES=['captured','spatial_nlm','fastdvdnet','fastdvdnet_55']

def credential():
    key=os.environ.get('GROQ_API_KEY')
    if not key and os.name=='nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as reg:key=winreg.QueryValueEx(reg,'GROQ_API_KEY')[0]
        except OSError:pass
    if not key:raise RuntimeError('GROQ_API_KEY is not visible in this process or the Windows user environment')
    return key

def request(session,method,path,**kwargs):
    for attempt in range(3):
        response=session.request(method,BASE+path,timeout=90,**kwargs)
        if response.status_code==429 and attempt<2:
            time.sleep(min(45,max(5,float(response.headers.get('retry-after','10')))));continue
        if not response.ok:
            try:message=str(response.json().get('error',{}).get('message','No message'))
            except ValueError:message='Non-JSON error body omitted'
            secret=session.headers.get('Authorization','').removeprefix('Bearer ')
            if secret:message=message.replace(secret,'[REDACTED]')
            raise RuntimeError(f'Groq HTTP {response.status_code}: {message[:800]}')
        return response.json()
    raise RuntimeError('Groq retries exhausted')

def tool(name,description,properties=None,required=None):
    return dict(type='function',function=dict(name=name,description=description,parameters=dict(type='object',properties=properties or {},required=required or [],additionalProperties=False)))

TOOLS=[tool('list_candidates','List available processing candidates.'),
    tool('inspect_candidate','Get measured blind image-quality scores and uncalibrated active-speaker outputs. These do not establish correctness.',{'candidate':{'type':'string','enum':NAMES}},['candidate']),
    tool('compare_with_capture','Compare candidate predictions and pixels with captured input. Disagreement is a risk signal, not a ground truth label.',{'candidate':{'type':'string','enum':NAMES}},['candidate']),
    tool('submit_selection','Select evidence for this one-second clip and decide whether the visible person speaks for the majority of the scored interval.',{'candidate':{'type':'string','enum':NAMES},'speaking':{'type':'boolean'},'confidence':{'type':'number','minimum':0,'maximum':1},'rationale':{'type':'string'}},['candidate','speaking','confidence','rationale'])]

class Environment:
    def __init__(self,case):
        # Explicit allowlist: ground-truth error/label/source/reference never stored here.
        self.data={k:{f:case['candidates'][k][f] for f in ['brisque','musiq','confidence','probability','visual_probability','input_mse']} for k in NAMES}
        self.submission=None
    def call(self,name,args):
        if name=='list_candidates':return {'candidates':NAMES,'captured_is_available':True,'frames_scored':24}
        candidate=args.get('candidate')
        if candidate not in NAMES:raise ValueError('Unknown candidate')
        v=self.data[candidate]
        if name=='inspect_candidate':return {k:v[k] for k in ('brisque','musiq','confidence','probability','visual_probability')}
        if name=='compare_with_capture':
            b=self.data['captured'];a=np.array(v['probability']);p=np.array(b['probability'])
            return dict(binary_prediction_disagreement=float(np.mean((a>=.5)!=(p>=.5))),mean_absolute_score_change=float(np.mean(abs(a-p))),input_pixel_mse=v['input_mse'])
        if name=='submit_selection':
            if type(args.get('speaking')) is not bool or not isinstance(args.get('confidence'),(int,float)) or not 0<=args['confidence']<=1 or not isinstance(args.get('rationale'),str):raise ValueError('Invalid submission')
            self.submission=args;return {'accepted':True}
        raise ValueError('Unknown tool')

def run(session,model,case):
    env=Environment(case)
    messages=[dict(role='system',content='You are an audiovisual evidence selection agent. Choose evidence that preserves the visible person\'s speaking activity while preferring good visual quality when safe. Use only supplied tools. BRISQUE lower is better; MUSIQ higher is better; neither measures attribution correctness. TalkNet scores are uncalibrated. Captured input can be noisy but restored input can alter speech evidence. Check captured input and alternatives, and compare with capture before selecting a restored candidate. Do not invent observations, ground truth, names, words, or calibrated confidence. Submit a selection within 8 model calls. You have no direct access to video/audio: your evidence comes from local measurements.'),dict(role='user',content='For this one-second segment, select captured or processed video as evidence and determine whether the visible person speaks for the majority of the interval. Explain briefly using the available measurements.')]
    usage=[];trace=[]
    for turn in range(8):
        data=request(session,'POST','/chat/completions',json=dict(model=model,messages=messages,tools=TOOLS,tool_choice='auto',temperature=0,max_completion_tokens=2048))
        msg=data['choices'][0]['message'];usage.append(data.get('usage',{}))
        messages.append({k:msg[k] for k in ('role','content','tool_calls') if k in msg})
        trace.append(dict(turn=turn,message=messages[-1]))
        calls=msg.get('tool_calls',[])
        if not calls:
            messages.append(dict(role='user',content='Complete the task using submit_selection.'))
        for call in calls[:16]:
            try:
                args=json.loads(call['function']['arguments']);output=env.call(call['function']['name'],args)
            except (ValueError,TypeError,KeyError) as e:output={'error':str(e)}
            response=dict(role='tool',tool_call_id=call['id'],content=json.dumps(output))
            messages.append(response);trace.append(response)
        if env.submission:return dict(status='completed',submission=env.submission,trace=trace,usage=usage)
    return dict(status='budget_exhausted',submission=None,trace=trace,usage=usage)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    session=requests.Session();session.headers['Authorization']='Bearer '+credential()
    models=request(session,'GET','/models')['data'];ids=sorted(x['id'] for x in models if 'qwen' in x['id'].lower() and x.get('active',True))
    requested=os.environ.get('GROQ_QWEN_MODEL')
    model=requested or ('qwen/qwen3.8-27b' if 'qwen/qwen3.8-27b' in ids else next(iter(ids),None))
    if model not in ids:raise RuntimeError('No requested Qwen model is available to this account; no non-Qwen substitution made')
    print(json.dumps({'available_qwen_models':ids,'selected':model}),flush=True)
    if args.check:return
    measurements=json.loads((ROOT/'results/agent_pilot_measurements.json').read_text())
    assert 'summary' in measurements,'Measurements must finish before agent evaluation'
    import hashlib
    mh=hashlib.sha256((ROOT/'results/agent_pilot_measurements.json').read_bytes()).hexdigest()
    path=ROOT/'results/qwen_agent_pilot.json'
    result=dict(model=model,measurements_sha256=mh,agent_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=[])
    if path.exists():
        result=json.loads(path.read_text());assert result['model']==model and result['measurements_sha256']==mh
    done={x['id'] for x in result['cases']}
    for case in measurements['cases']:
        if case['id'] in done:continue
        record=run(session,model,case);record['id']=case['id']
        # Evaluation only after final action; never fed back to agent.
        if record['submission']:
            s=record['submission'];record['evaluation']=dict(selected_frame_error=case['candidates'][s['candidate']]['error'],majority_answer_correct=s['speaking']==bool(case['label']),captured_frame_error=case['candidates']['captured']['error'])
        result['cases'].append(record);path.write_text(json.dumps(result,indent=2))
        print(case['id'],record['status'],record.get('evaluation'),flush=True)

if __name__=='__main__':main()

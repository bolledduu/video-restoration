"""Bounded native audiovisual Gemini pilot; not the CTS framework or tool-agent test."""
import argparse,base64,hashlib,json,os,random,subprocess,time
from pathlib import Path
import numpy as np
import requests

MODELS=['gemini-3.5-flash-lite','gemini-2.5-flash-lite']
CONDITIONS=['captured','fastdvdnet']
PROMPT='''This is a four-second face-crop video with synchronized scene audio. Determine whether the visible person is audibly speaking, not merely whether speech is present in the scene. Another person or an offscreen person may produce the audio. Do not identify anyone or infer commitments or words. For each interval [0,1), [1,2), [2,3), [3,4) seconds, decide whether this visible person is audibly speaking for MORE THAN HALF of that interval. Return four decisions in order: 1=speaking, 0=not speaking, -1=insufficient evidence. Also give your best binary majority_speaking decision (0 or 1) for MORE THAN HALF of the entire four seconds, even if uncertain. Do not return frame-level claims or explanations. Return only the requested JSON.'''
SCHEMA={'type':'OBJECT','properties':{'decisions':{'type':'ARRAY','items':{'type':'INTEGER'},'minItems':4,'maxItems':4},'majority_speaking':{'type':'INTEGER'}},'required':['decisions','majority_speaking']}
CONFIG={'temperature':0,'maxOutputTokens':1024,'responseMimeType':'application/json','responseSchema':SCHEMA}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,obj):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
def validate_answer(answer):
    if not isinstance(answer,dict) or set(answer)!={'decisions','majority_speaking'}:raise ValueError('Unexpected response schema')
    values=answer['decisions']
    if not isinstance(values,list) or len(values)!=4 or any(type(x) is not int or x not in (-1,0,1) for x in values):raise ValueError('Invalid interval decisions')
    if type(answer['majority_speaking']) is not int or answer['majority_speaking'] not in (0,1):raise ValueError('Invalid majority decision')
    return answer

def interval_truth(labels):
    labels=np.asarray(labels)
    if labels.shape!=(100,) or not np.isin(labels,[0,1]).all():raise ValueError('Require exactly 100 official binary labels at 25 fps')
    return [int(np.sum(labels[i:i+25])>12.5) for i in range(0,100,25)],int(np.sum(labels)>50)

def score_answer(answer,truth,majority):
    validate_answer(answer)
    answered=[i for i,x in enumerate(answer['decisions']) if x!=-1]
    wrong=sum(answer['decisions'][i]!=truth[i] for i in answered)
    return dict(eligible=4,answered=len(answered),errors=wrong,error_rate=wrong/len(answered) if answered else None,
                majority_error=int(answer['majority_speaking']!=majority))

def paired_counts(captured,restored,truth):
    validate_answer(captured);validate_answer(restored)
    both=[i for i in range(4) if captured['decisions'][i]!=-1 and restored['decisions'][i]!=-1]
    return dict(jointly_answered=len(both),
        captured_errors=sum(captured['decisions'][i]!=truth[i] for i in both),
        restored_errors=sum(restored['decisions'][i]!=truth[i] for i in both),
        harm=sum(captured['decisions'][i]==truth[i] and restored['decisions'][i]!=truth[i] for i in both),
        corrections=sum(captured['decisions'][i]!=truth[i] and restored['decisions'][i]==truth[i] for i in both))

def make_payload(media):
    # No case names, conditions, labels, quality scores or TalkNet values leave this boundary.
    return {'contents':[{'role':'user','parts':[{'inlineData':{'mimeType':'video/mp4','data':base64.b64encode(Path(media).read_bytes()).decode()},'videoMetadata':{'fps':24}}, {'text':PROMPT}]}], 'generationConfig':CONFIG}

class Gemini:
    def __init__(self):
        self.key=os.environ['GEMINI_API_KEY'];self.backup=os.environ.get('GEMINI_API_KEY_BACKUP');self.backup_used=False
        self.session=requests.Session()
    def request(self,method,path,payload=None):
        for attempt in range(2):
            try:r=self.session.request(method,'https://generativelanguage.googleapis.com/v1beta/'+path,
                    headers={'x-goog-api-key':self.key},json=payload,timeout=90)
            except requests.RequestException:raise RuntimeError('Transport error; request outcome unknown; not automatically retried') from None
            if r.ok:return r.json()
            # No rotation for quota/rate errors; no hidden retries of billable requests.
            if r.status_code==401 and self.backup and not self.backup_used:
                self.key=self.backup;self.backup_used=True;continue
            raise RuntimeError('Google HTTP '+str(r.status_code)+'; response withheld to avoid credential disclosure')
        raise RuntimeError('Authentication failed')

def export_media(bundle,condition,audio_path,out,ffmpeg):
    video=bundle[condition]
    if video.shape!=(100,128,128,3) or video.dtype!=np.uint8:raise ValueError('Unexpected video array')
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','128x128','-r','25','-i','pipe:0',
         '-i',str(audio_path),'-map','0:v:0','-map','1:a:0','-c:v','libx264rgb','-crf','0','-preset','fast','-threads','2',
         '-c:a','aac','-b:a','64k','-t','4','-movflags','+faststart',str(out)]
    subprocess.run(cmd,input=video.tobytes(),capture_output=True,check=True,timeout=90)
    decoded=subprocess.run([ffmpeg,'-v','error','-i',str(out),'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True,timeout=60).stdout
    if decoded!=video.tobytes():raise ValueError('Video encoding changed candidate pixels')
    decoded_audio=subprocess.run([ffmpeg,'-v','error','-i',str(out),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','pipe:1'],capture_output=True,check=True,timeout=60).stdout
    if len(decoded_audio)//4<64000:raise ValueError('Audio shorter than clip')
    return dict(path=str(out.resolve()),sha256=sha(out),decoded_pixels_sha256=hashlib.sha256(decoded).hexdigest(),decoded_audio_sha256=hashlib.sha256(decoded_audio).hexdigest(),decoded_audio_samples=len(decoded_audio)//4)

def prepare(args):
    from scipy.io import wavfile
    if args.output.exists():raise FileExistsError('Use a new output directory; preserve previous attempts')
    args.output.mkdir(parents=True);media_dir=args.media_dir.resolve();media_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for measurement_path in args.measurements:
        data=json.loads(measurement_path.read_text())
        if data['status']!='COMPLETE':raise ValueError('Incomplete input measurements')
        for case in data['cases']:
            bundle_path=Path(case['local_bundle'])
            if sha(bundle_path)!=case['bundle_sha256']:raise ValueError('Bundle hash mismatch')
            with np.load(bundle_path,allow_pickle=False) as bundle:
                truth,majority=interval_truth(bundle['labels'])
                audio=bundle['audio']
                if audio.shape!=(64000,) or not np.isfinite(audio).all():raise ValueError('Invalid audio')
                wav=media_dir/(case['id']+'.wav')
                if wav.exists():raise FileExistsError(wav)
                wavfile.write(wav,16000,audio.astype(np.float32))
                candidates={}
                for condition in CONDITIONS:
                    candidates[condition]=export_media(bundle,condition,wav,media_dir/(case['id']+'_'+condition+'.mp4'),args.ffmpeg)
                if len({c['decoded_audio_sha256'] for c in candidates.values()})!=1:raise ValueError('Candidate audio changed')
            rows.append(dict(id=case['id'],source=case['source'],start=case['start'],truth=truth,majority_truth=majority,
                input_measurement=str(measurement_path),input_measurement_sha256=sha(measurement_path),bundle_sha256=case['bundle_sha256'],
                audio_original_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),media=candidates,
                quality={k:{m:case['candidates'][k][m] for m in ('musiq','brisque')} for k in CONDITIONS}))
    if len(rows)!=4 or len({r['id'] for r in rows})!=4:raise ValueError('Frozen pilot requires exactly four distinct episodes')
    jobs=[]
    for model in MODELS:
        for repeat in range(2):
            for row in rows:
                for condition in CONDITIONS:jobs.append(dict(model=model,repeat=repeat,case_id=row['id'],condition=condition))
    random.Random(20261017).shuffle(jobs)
    protocol=dict(status='PREPARED',scope='Exploratory native audiovisual perception pilot; not held-out confirmation, agent tooling, or CTS evaluation',
        models=MODELS,conditions=CONDITIONS,repetitions=2,prompt=PROMPT,generation_config=CONFIG,requested_video_fps=24,
        output_unit='One-second majority speaking decisions plus clip majority; never dense frame accuracy',
        label_rule='More than half of 25 official frame labels per second; clip tie maps to not speaking',
        audio_policy='Identical AAC encoding across conditions, decoded equality verified; originals preserved',
        cost_limit_usd=1.0,cost_reservation='CountTokens plus 1024 output; conservative rates $0.30/M input and $2.50/M output for both models; not a billing receipt',
        selection='All four existing hashed AVA development bundles; not selected by Gemini outcomes',
        code_sha256=sha(__file__),cases=rows,jobs=jobs,
        limitations=['Two source videos only','Existing exploratory clips','Official labels without independent relabeling','Requested FPS does not prove internal model sampling','Single-person crop omits competing faces'])
    save(args.output/'protocol.json',protocol);print('Prepared',len(rows),'cases and',len(jobs),'requests',flush=True)

def run(args):
    protocol_path=args.output/'protocol.json';protocol=json.loads(protocol_path.read_text())
    if protocol['code_sha256']!=sha(__file__):raise ValueError('Code changed after protocol freeze')
    if protocol['prompt']!=PROMPT or protocol['generation_config']!=CONFIG:raise ValueError('Prompt/config mismatch')
    output=args.output/'runs.json'
    if output.exists():raise FileExistsError('Run already exists; no automatic repeated billing')
    api=Gemini();available=api.request('GET','models')
    model_names={m['name'] for m in available['models']}
    if any('models/'+m not in model_names for m in MODELS):raise RuntimeError('Exact model unavailable; no substitution')
    save(args.output/'availability.json',{'requested_models':[m for m in available['models'] if m['name'] in {'models/'+v for v in MODELS}]})
    result=dict(status='RUNNING',protocol_sha256=sha(protocol_path),reserved_upper_cost_usd=0.0,runs=[])
    save(output,result);cases={c['id']:c for c in protocol['cases']};token_counts={}
    for job in protocol['jobs']:
        case=cases[job['case_id']];media=case['media'][job['condition']]
        if sha(media['path'])!=media['sha256']:raise ValueError('Media changed')
        payload=make_payload(media['path']);row={**job,'media_sha256':media['sha256'],'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
        try:
            key=(job['model'],media['sha256'])
            if key not in token_counts:
                token_counts[key]=api.request('POST','models/'+job['model']+':countTokens',{'contents':payload['contents']})['totalTokens']
            count=token_counts[key];reserve=count*.30/1e6+1024*2.50/1e6
            if count>50000 or result['reserved_upper_cost_usd']+reserve>protocol['cost_limit_usd']:raise RuntimeError('Pilot cost/token ceiling reached')
            row.update(status='REQUEST_PENDING',input_token_count=count,reserved_upper_cost_usd=reserve)
            result['reserved_upper_cost_usd']+=reserve;result['runs'].append(row);save(output,result)
            response=api.request('POST','models/'+job['model']+':generateContent',payload)
            row['response']=response
            candidates=response.get('candidates',[])
            if len(candidates)!=1 or candidates[0].get('finishReason')!='STOP':raise ValueError('Response did not complete normally')
            text=''.join(part.get('text','') for part in candidates[0]['content']['parts'] if not part.get('thought'))
            answer=validate_answer(json.loads(text));row.update(status='COMPLETE',answer=answer,evaluation=score_answer(answer,case['truth'],case['majority_truth']))
            save(output,result);print(job['model'],job['case_id'],job['condition'],job['repeat'],row['evaluation'],flush=True)
        except (RuntimeError,ValueError,KeyError) as error:
            row.update(status='FAILED',error=str(error))
            if not result['runs'] or result['runs'][-1] is not row:result['runs'].append(row)
            result['status']='INCOMPLETE';save(output,result);raise SystemExit('Pilot stopped: '+str(error)) from None
        time.sleep(3)
    result['status']='COMPLETE';save(output,result)

def summarize(args):
    protocol=json.loads((args.output/'protocol.json').read_text());runs=json.loads((args.output/'runs.json').read_text())
    if runs['status']!='COMPLETE':raise ValueError('Cannot summarize incomplete pilot as complete')
    cases={c['id']:c for c in protocol['cases']};rows=[]
    for model in MODELS:
        for repeat in range(2):
            for cid,case in cases.items():
                pair={r['condition']:r for r in runs['runs'] if r['model']==model and r['repeat']==repeat and r['case_id']==cid and r['status']=='COMPLETE'}
                if set(pair)!=set(CONDITIONS):raise ValueError('Missing paired run')
                c=pair['captured']['answer'];r=pair['fastdvdnet']['answer']
                rows.append(dict(model=model,repeat=repeat,case_id=cid,source=case['source'],truth=case['truth'],captured=c,restored=r,
                    **paired_counts(c,r,case['truth']),majority_captured_error=int(c['majority_speaking']!=case['majority_truth']),majority_restored_error=int(r['majority_speaking']!=case['majority_truth'])))
    save(args.output/'paired_summary.json',dict(scope=protocol['scope'],paired_rows=rows,source_count=len({c['source'] for c in cases.values()}),
        note='Repetitions and bins are dependent; descriptive counts only; no population inference'))
    for model in MODELS:
        subset=[r for r in rows if r['model']==model]
        print(model,{k:sum(r[k] for r in subset) for k in ('jointly_answered','captured_errors','restored_errors','harm','corrections','majority_captured_error','majority_restored_error')})

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','summarize']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--measurements',type=Path,nargs='+');p.add_argument('--media-dir',type=Path);p.add_argument('--ffmpeg');args=p.parse_args()
    {'prepare':prepare,'run':run,'summarize':summarize}[args.stage](args)
if __name__=='__main__':main()

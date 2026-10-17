"""Frozen CPU study: actual audit training, independent calibration, untouched-source testing."""
import argparse,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path

SCRIPT_NAMES=['run_ava_local.py','talknet_checkpoint_cpu.py','captured_support.py','measure_captured_support.py','evaluate_captured_support.py','captured_support_tool.py','ava_evidence_environment.py']

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def read(p):return json.loads(Path(p).read_text())
def normalize_training(protocol,paths,checkpoint):
    roster={(c['source'],c['entity'],round(c['start'],5)):c for c in protocol['cases'] if c['split']=='train'};cases=[]
    for path in paths:
        d=read(path)
        if d['status'] not in ('COMPLETE','COMPLETE_WITH_EXCLUSIONS'):raise ValueError('Incomplete training input')
        weight=d.get('detector_checkpoint_sha256',d.get('weights',{}).get('talknet'))
        if weight!=checkpoint:raise ValueError('Training detector checkpoint mismatch')
        for c in d['cases']:
            key=(c['source'],c['entity'],round(c['start'],5))
            if key not in roster:raise ValueError('Unplanned training episode')
            c={**c,'original_case_id':c['id'],'id':roster[key]['id']};cases.append(c)
    if len(cases)!=len(roster) or {c['id'] for c in cases}!={c['id'] for c in roster.values()}:raise ValueError('Training roster incomplete or duplicated')
    return dict(status='COMPLETE',detector_checkpoint_sha256=checkpoint,cases=cases,parent_measurements={str(p):sha(p) for p in paths})

def serialized(model):
    return dict(classes=model['classifier'].classes_.tolist(),coef=model['classifier'].coef_.tolist(),intercept=model['classifier'].intercept_.tolist(),mean=model['scaler'].mean_.tolist(),scale=model['scaler'].scale_.tolist())
def infer(model,case,tau):
    import numpy as np
    from evaluate_captured_support import extract
    x,pos=extract(case);mean=np.asarray(model['mean']);scale=np.asarray(model['scale']);z=((x[:,:len(mean)]-mean)/scale)@np.asarray(model['coef']).T+np.asarray(model['intercept'])
    z-=z.max(axis=1,keepdims=True);p=np.exp(z);p/=p.sum(axis=1,keepdims=True)
    h=float(p[:,1].mean());g=float(p[:,2].mean());q=case['quality'][case['candidate_key']]['musiq']>case['quality']['captured']['musiq']
    return bool(q and h<=tau and g>=h),p,pos

def learning_stage(args):
    import numpy as np
    from evaluate_captured_support import fit,evaluate,transition_counts,audit_grid_transition_counts
    job=args.job;config=read(job/'config.json');protocol=read(job/'source_protocol.json');split=args.stage
    allowed=protocol['splits']
    def load_part(name):
        d=read(job/name/'audit/measurements.json')
        if d['status']!='COMPLETE':raise ValueError('Incomplete '+name+' audit')
        if d['weights']['talknet']!=config['checkpoint_sha256']:raise ValueError('Detector changed')
        if any(c['source'] not in allowed[name] for c in d['cases']):raise ValueError('Split source leakage')
        return d['cases']
    if split=='fit':
        cases=load_part('train');models={v:serialized(fit(cases,v)) for v in ('base','ordinary_audit','support')}
        save(job/'trained.json',dict(status='TRAINED',models=models,training_sources=allowed['train'],training_measurements_sha256=sha(job/'train/audit/measurements.json'),transition_counts=transition_counts(cases),audit_grid_transitions=audit_grid_transition_counts(cases)))
    elif split=='calibrate':
        artifact=read(job/'trained.json');cases=load_part('calibration');calibrated={}
        for name,model in artifact['models'].items():
            grid=[]
            for tau in (-1.,0.,.01,.02,.05,.1,.2,.5,1.):
                rows=[evaluate(c,infer(model,c,tau)[0]) for c in cases]
                grid.append(dict(tau=tau,harm=sum(r['harm'] for r in rows),gain=sum(r['corrections']-r['harm'] for r in rows),quality=sum(r['delta_musiq'] for r in rows)))
            feasible=[v for v in grid if v['harm']==0 and v['gain']>=0];chosen=max(feasible,key=lambda v:(v['gain'],v['quality'],-v['tau']))
            calibrated[name]=dict(model=model,tau=chosen['tau'],grid=grid)
        save(job/'frozen_selector.json',dict(status='FROZEN_BEFORE_TEST',variants=calibrated,trained_sha256=sha(job/'trained.json'),calibration_sha256=sha(job/'calibration/audit/measurements.json'),calibration_transitions=transition_counts(cases),note='Zero observed calibration harm is an empirical constraint, not a statistical safety guarantee'))
    elif split=='evaluate':
        frozen=read(job/'frozen_selector.json');cases=load_part('test');rows={}
        for name,item in frozen['variants'].items():rows[name]=[evaluate(c,infer(item['model'],c,item['tau'])[0]) for c in cases]
        rows['captured']=[evaluate(c,False) for c in cases];rows['restored']=[evaluate(c,True) for c in cases]
        rows['musiq']=[evaluate(c,c['quality'][c['candidate_key']]['musiq']>c['quality']['captured']['musiq']) for c in cases]
        rows['brisque']=[evaluate(c,c['quality'][c['candidate_key']]['brisque']<c['quality']['captured']['brisque']) for c in cases]
        rows['quality_consensus']=[evaluate(c,c['quality'][c['candidate_key']]['musiq']>c['quality']['captured']['musiq'] and c['quality'][c['candidate_key']]['brisque']<c['quality']['captured']['brisque']) for c in cases]
        rows['confidence']=[evaluate(c,float(np.mean(abs(np.asarray(c['restored_probability'])-.5)))>float(np.mean(abs(np.asarray(c['captured_probability'])-.5)))) for c in cases]
        rows['disagreement_fallback']=[evaluate(c,False) for c in cases]
        summaries={name:{k:sum(r[k] for r in values) for k in ('eligible','errors','captured_errors','harm','corrections')} for name,values in rows.items()}
        for name,values in rows.items():summaries[name].update(mean_delta_musiq=float(np.mean([r['delta_musiq'] for r in values])),restored_cases=sum(r['selected']=='restored' for r in values))
        sources=sorted({c['source'] for c in cases});rng=np.random.default_rng(20261017);cis={}
        for name,values in rows.items():
            groups={s:[r for r in values if r['source']==s] for s in sources};boot=[]
            for _ in range(2000):
                selected=[r for s in rng.choice(sources,len(sources),replace=True) for r in groups[s]]
                boot.append(sum(r['captured_errors']-r['errors'] for r in selected)/sum(r['eligible'] for r in selected))
            cis[name]=np.percentile(boot,[2.5,97.5]).tolist()
        from captured_support_tool import assess
        raw_test=read(job/'test/raw/measurements.json')
        agent_cases=[{**c,'candidates':{k:v for k,v in c['candidates'].items() if k in ('captured','fastdvdnet')}} for c in raw_test['cases']]
        save(job/'agent_measurements.json',dict(status='COMPLETE',scope='Held-out tool-agent inputs; labels retained for offline scoring only',cases=agent_cases))
        values={c['id']:assess(c,frozen['variants']['support']['model'],frozen['variants']['support']['tau'],sha(job/'frozen_selector.json')) for c in cases}
        save(job/'assessments.json',dict(measurements_sha256=sha(job/'agent_measurements.json'),model_artifact=str(job/'frozen_selector.json'),model_artifact_sha256=sha(job/'frozen_selector.json'),assessments=values))
        support=summaries['support'];competitors=['captured','musiq','brisque','quality_consensus','base','ordinary_audit'];best=min(summaries[v]['errors'] for v in competitors)
        verdict='OBSERVED_ADVANTAGE_REQUIRES_UNCERTAINTY_AND_TRANSFER_REVIEW' if support['errors']<best else 'NO_OBSERVED_ATTRIBUTION_ADVANTAGE_OVER_STRONGEST_BASELINE'
        save(job/'TEST_RESULTS.json',dict(status='COMPLETE',scope='Primary AVA sigma25 FastDVDnet/TalkNet-AVA deterministic selector evaluation; one dataset and restoration family',frozen_selector_sha256=sha(job/'frozen_selector.json'),test_audit_sha256=sha(job/'test/audit/measurements.json'),source_count=len(sources),case_count=len(cases),test_exclusions=raw_test.get('exclusions',[]),summaries=summaries,source_bootstrap_accuracy_gain_ci=cis,rows=rows,verdict=verdict,limits=['Sparse IQA measurements; no human perceptual-quality claim','No real LLM-agent comparison in this CPU study','No unseen-restorer or detector-transfer claim','Lower-face rectangular proxy; not verified mouth landmarks','Clipped/quantized noise: empirical measurement, not unbiased certificate'] ))
        lines=['# Primary corrected AVA experiment','',verdict,'','| Method | Errors / eligible | Harms | Corrections | Restored clips | Mean MUSIQ change |','|---|---:|---:|---:|---:|---:|']
        for name,v in summaries.items():lines.append(f"| {name} | {v['errors']}/{v['eligible']} | {v['harm']} | {v['corrections']} | {v['restored_cases']} | {v['mean_delta_musiq']:.3f} |")
        lines+=['',f'Sources: {len(sources)}; clips: {len(cases)}. All methods answer all timestamps. Confidence intervals and raw per-source rows are in TEST_RESULTS.json.','', 'This completes the primary deterministic study. It does not establish a successful agent framework, human preference, novelty or conference readiness. No LaTeX was edited.']
        (job/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main_job(args):
    job=args.job;config=read(job/'config.json');protocol=read(job/'source_protocol.json');scripts=job/'code';state_path=job/'STATE.json'
    def state(phase,status='RUNNING',**extra):
        record=dict(status=status,phase=phase,updated_unix=time.time(),**extra)
        save(state_path,record)
        with (job/'EVENTS.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    def run(command):
        subprocess.run([sys.executable,*map(str,command)],check=True,cwd=config['repo'])
    def verify_code():
        for name,digest in config['code_sha256'].items():
            if sha(scripts/name)!=digest:raise ValueError('Frozen code changed: '+name)
    def create_split(name):
        root=job/name/'data';root.mkdir(parents=True,exist_ok=True);cases=[c for c in protocol['cases'] if c['split']==name];sources=protocol['splits'][name]
        save(root/'manifest.json',dict(selected_sources=sources,episodes=[{**c,'episode_id':c['id']} for c in cases],parent_protocol_sha256=sha(job/'source_protocol.json')))
        receipts=[]
        for source in sources:
            path=next(c['media_path'] for c in cases if c['source']==source);receipts.append(dict(source=source,path=path,sha256=sha(Path(path))))
        save(root/'media_download_audit.json',receipts);return root
    def audit(name,measurements):
        run([scripts/'measure_captured_support.py','--runtime-root',config['runtime'],'--output',job/name/'audit','--measurement-files',measurements,'--detector-checkpoint',config['checkpoint'],'--compact-controls'])
    try:
        verify_code();state('train_audit');audit('train',job/'train/normalized_measurements.json')
        state('fit_training')
        if not (job/'trained.json').exists():run([job/'code/run_primary_cts_study.py','--stage','fit','--job',job])
        state('calibration_raw');data=create_split('calibration')
        run([scripts/'run_ava_local.py','--runtime-root',config['runtime'],'--data-root',data,'--max-sources','5','--detector-checkpoint',config['checkpoint'],'--output',job/'calibration/raw'])
        state('calibration_audit');audit('calibration',job/'calibration/raw/measurements.json')
        state('freeze_calibration')
        if not (job/'frozen_selector.json').exists():run([job/'code/run_primary_cts_study.py','--stage','calibrate','--job',job])
        frozen_hash=sha(job/'frozen_selector.json');state('test_raw',frozen_selector_sha256=frozen_hash);data=create_split('test')
        run([scripts/'run_ava_local.py','--runtime-root',config['runtime'],'--data-root',data,'--max-sources','15','--detector-checkpoint',config['checkpoint'],'--output',job/'test/raw'])
        state('test_audit',frozen_selector_sha256=frozen_hash);audit('test',job/'test/raw/measurements.json')
        if sha(job/'frozen_selector.json')!=frozen_hash:raise ValueError('Selector changed after test began')
        state('evaluate_test');run([job/'code/run_primary_cts_study.py','--stage','evaluate','--job',job])
        state('finished','COMPLETE',results=str(job/'TEST_RESULTS.json'),report=str(job/'REPORT.md'))
    except Exception as error:
        state('stopped','FAILED',error_type=type(error).__name__,error=str(error));traceback.print_exc();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--stage',choices=['run','fit','calibrate','evaluate'],default='run');args=p.parse_args()
    if args.stage=='run':main_job(args)
    else:learning_stage(args)

"""Measure the actual masked audit with real FastDVDnet and frozen TalkNet on CPU."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from captured_support import MASK_SEEDS,SHIFTS,region_mask,fixed_mask,masked_fill,support_scores,quantize,logits,transition_targets

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)

def collect(pilot):
    rows=[]
    for name in ('transfer_rq_measurements','benefit_measurements'):
        path=pilot/'results'/f'{name}.json';doc=json.loads(path.read_text())
        for c in doc['cases']:
            if name=='benefit_measurements' and not c['id'].endswith('__noise25'):continue
            key='fastdvdnet' if name=='transfer_rq_measurements' else 'restored'
            rows.append(dict(id=c['id'],source=c['source'],source_bundle=c['path'],source_sha256=c['sha256'],
                output_bundle=c['outputs'],output_sha256=c['output_sha256'],candidate_key=key,
                quality={k:{m:c['candidates'][k][m] for m in ('musiq','brisque')} for k in ('captured',key)},measurement_sha256=sha(path)))
    for name in ('ava_smoke_v1','ava_corrected_dev_v3'):
        path=pilot/'results'/name/'measurements.json';doc=json.loads(path.read_text())
        for c in doc['cases']:
            rows.append(dict(id=c['id'],source=c['source'],source_bundle=c['local_bundle'],source_sha256=c['bundle_sha256'],
                output_bundle=c['local_bundle'],output_sha256=c['bundle_sha256'],candidate_key='fastdvdnet',
                quality={k:{m:c['candidates'][k][m] for m in ('musiq','brisque')} for k in ('captured','fastdvdnet')},measurement_sha256=sha(path)))
    return rows

def collect_measured_inputs(files):
    rows=[]
    for path in files:
        document=json.loads(Path(path).read_text())
        if document['status'] not in ('COMPLETE','COMPLETE_WITH_EXCLUSIONS'):raise ValueError('Incomplete measured input')
        for c in document['cases']:
            rows.append(dict(id=c['id'],source=c['source'],source_bundle=c['local_bundle'],source_sha256=c['bundle_sha256'],output_bundle=c['local_bundle'],output_sha256=c['bundle_sha256'],candidate_key='fastdvdnet',quality={k:{m:c['candidates'][k][m] for m in ('musiq','brisque')} for k in ('captured','fastdvdnet')},measurement_sha256=sha(Path(path))))
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate measured cases')
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--measurement-files',type=Path,nargs='+');ap.add_argument('--detector-checkpoint',type=Path);ap.add_argument('--compact-controls',action='store_true');args=ap.parse_args()
    pilot=Path(__file__).resolve().parents[1];root=args.runtime_root.resolve();out=args.output;out.mkdir(parents=True,exist_ok=True)
    rows=collect_measured_inputs(args.measurement_files) if args.measurement_files else collect(pilot)
    protocol=dict(scope='Actual CTS component development on all available noise25 pairs in predefined cohorts; historical sources only; NOT final validation',
        rows=rows,mask_seeds=MASK_SEEDS,mask_fraction=.10,mask_type='Fixed spatial sites across all frames; all RGB channels withheld',
        shifts=SHIFTS,region_definition='Prespecified normalized lower-face proxy, not verified mouth landmarks; no full-input registration',
        degradation='Already cached clipped quantized Gaussian sigma25; empirical scores, no unbiased-risk claim',
        audio_policy='Unchanged original audio for every detector call',code_sha256=sha(__file__),core_sha256=sha(Path(__file__).with_name('captured_support.py')),
        control_policy='Support MSE: lower-face, cheek, full frame, static, half/double dynamic amplitude; detector: shifts for each region and static/amplitude for lower face')
    if args.measurement_files:protocol['scope']='Real masked audits on supplied measured AVA split; split membership and frozen model are checked by the study orchestrator'
    protocol['compact_controls']=args.compact_controls
    if args.compact_controls:protocol['control_policy']='Lower-face timing, static and amplitude ASD probes; cheek/full-frame support MSE controls. Cheek/full-frame ASD effects are not measured.'
    if args.detector_checkpoint:protocol['detector_checkpoint_sha256']=sha(args.detector_checkpoint)
    pp=out/'protocol.json'
    if pp.exists():
        if json.loads(pp.read_text())!=json.loads(json.dumps(protocol)):raise ValueError('Resume configuration mismatch')
    else:save(pp,protocol)
    for folder in ('tools/quality_cpu','tools/iqa_models','tools/python','scripts','external/fastdvdnet'):sys.path.insert(0,str(root/folder))
    import torch,cv2
    from models import FastDVDnet
    from fastdvdnet import denoise_seq_fastdvdnet
    from evaluate_causal import TalkNetCPU
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.manual_seed(0)
    model=FastDVDnet().cpu().eval();weight=root/'external/fastdvdnet/model_clipped_noise.pth'
    state=torch.load(weight,map_location='cpu',weights_only=True);model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()})
    if args.detector_checkpoint:
        from talknet_checkpoint_cpu import load_detector
        detector=load_detector(root,args.detector_checkpoint)
    else:detector=TalkNetCPU()
    weights=dict(fastdvdnet=sha(weight),talknet=detector.checkpoint_sha256)
    rp=out/'measurements.json'
    result=json.loads(rp.read_text()) if rp.exists() else dict(status='RUNNING',protocol_sha256=sha(pp),weights=weights,cases=[])
    if result['protocol_sha256']!=sha(pp) or result['weights']!=weights:raise ValueError('Resume identity mismatch')
    save(rp,result)
    for row in rows:
        if row['id'] in {c['id'] for c in result['cases']}:continue
        started=time.monotonic()
        for path,key in ((row['source_bundle'],'source_sha256'),(row['output_bundle'],'output_sha256')):
            if sha(path)!=row[key]:raise ValueError('Input changed')
        with np.load(row['source_bundle'],allow_pickle=False) as z:audio=z['audio'].copy();labels=z['labels'].copy()
        with np.load(row['output_bundle'],allow_pickle=False) as z:captured=z['captured'].copy();full=z[row['candidate_key']].copy()
        T,H,W,_=captured.shape;boxes=[[0,0,W,H]]*T;audio_hash=hashlib.sha256(audio.tobytes()).hexdigest()
        def predict(v):
            pred,indices=detector.predict(quantize(v),audio,boxes,25)
            return pred,indices
        cp,indices=predict(captured);fp,fi=predict(full)
        np.testing.assert_array_equal(indices,fi);np.testing.assert_array_equal(cp['A'],fp['A'])
        # Confirm the deployed checkpoint matches the cached full candidate.
        with torch.inference_mode():
            reproduced=denoise_seq_fastdvdnet(torch.from_numpy(captured.transpose(0,3,1,2).copy()).float()/255,torch.tensor([25/255]),5,model)
        reproduced=quantize(reproduced.permute(0,2,3,1).numpy()*255)
        if not np.array_equal(reproduced,full):raise ValueError('Cached candidate differs from deployed FastDVDnet; do not transfer masked scores')
        folds=[]
        for seed in MASK_SEEDS:
            hidden=fixed_mask(H,W,seed);base=quantize(masked_fill(captured,hidden))
            with torch.inference_mode():
                restored=denoise_seq_fastdvdnet(torch.from_numpy(base.transpose(0,3,1,2).copy()).float()/255,torch.tensor([25/255]),5,model)
            restored=quantize(restored.permute(0,2,3,1).numpy()*255)
            factual,idx=predict(restored);np.testing.assert_array_equal(indices,idx);np.testing.assert_array_equal(cp['A'],factual['A'])
            record=dict(seed=seed,masked_input_sha256=hashlib.sha256(base.tobytes()).hexdigest(),audit_output_sha256=hashlib.sha256(restored.tobytes()).hexdigest(),
                audit_probability=factual['AV'].tolist(),full_audit_pixel_mse=float(np.mean(((full.astype(float)-restored)/255)**2)),regions={})
            for region_name in ('lower_face','cheek','full_frame'):
                region=region_mask(H,W,region_name);support,variants=support_scores(captured,base,restored,hidden,region)
                region_record=dict(indices=support['indices'].tolist(),support=support['temporal_support'].tolist(),
                    mse={k:v.tolist() for k,v in support['mse'].items()},probe_probabilities={},quantization_mse={})
                names=['shift_'+str(s) for s in SHIFTS] if region_name=='lower_face' or not args.compact_controls else []
                if region_name=='lower_face':names+=['static','scale_0.5','scale_1.5']
                for name in names:
                    probe=variants[name];pred,idx=predict(probe);np.testing.assert_array_equal(indices,idx);np.testing.assert_array_equal(cp['A'],pred['A'])
                    region_record['probe_probabilities'][name]=pred['AV'].tolist()
                    region_record['quantization_mse'][name]=float(np.mean(((quantize(probe).astype(float)-probe)/255)**2))
                record['regions'][region_name]=region_record
            folds.append(record)
        if hashlib.sha256(audio.tobytes()).hexdigest()!=audio_hash:raise ValueError('Audio mutated')
        target=transition_targets(labels[indices],cp['AV'],fp['AV'])
        case=dict(id=row['id'],source=row['source'],audio_sha256=audio_hash,indices=indices.tolist(),labels=labels[indices].tolist(),
            captured_probability=cp['AV'].tolist(),restored_probability=fp['AV'].tolist(),transition_targets=target.tolist(),quality=row['quality'],candidate_key=row['candidate_key'],folds=folds,
            elapsed_seconds=time.monotonic()-started,region_caveat='Lower-face proxy only; no verified mouth landmark claim')
        result['cases'].append(case);save(rp,result)
        print(row['id'],'transition counts',np.bincount(target,minlength=4).tolist(),'seconds',round(case['elapsed_seconds']),flush=True)
    result['status']='COMPLETE';save(rp,result)
if __name__=='__main__':main()

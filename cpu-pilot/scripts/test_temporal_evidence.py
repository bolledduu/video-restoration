"""Fixed exploratory mechanism screen; no labels used in prototype construction."""
import argparse
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from localize_restoration_effects import sha,save,stats

P=Path(__file__).resolve().parents[1]

def construct(c,r):
    c=c.astype(float);r=r.astype(float);residual=r-c;mean=residual.mean(axis=0,keepdims=True)
    raw={'static_residual':c+mean,'dynamic_residual':c+residual-mean,'half_blend':(c+r)/2}
    assert np.allclose(np.diff(raw['static_residual'],axis=0),np.diff(c,axis=0))
    assert np.allclose(np.diff(raw['dynamic_residual'],axis=0),np.diff(r,axis=0))
    return {k:np.rint(v.clip(0,255)).astype(np.uint8) for k,v in raw.items()}, {k:float(np.mean((v<0)|(v>255))) for k,v in raw.items()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime-root',type=Path,required=True);root=parser.parse_args().runtime_root.resolve()
    rows=[];hashes={}
    for name in ['transfer_rq_measurements','benefit_measurements']:
        file=P/'results'/f'{name}.json';d=json.loads(file.read_text());assert d['status']=='COMPLETE';hashes[name]=sha(file)
        for case in d['cases']:
            for candidate in (['fastdvdnet','spatial_nlm'] if name=='transfer_rq_measurements' else ['restored']):
                rows.append(dict(id=case['id']+'__'+candidate,case=case,candidate=candidate))
    protocol={'scope':'Exploratory engineering on all 17 previously observed pairs; not held-out validation. Independent 12-source queue remains untouched.',
        'created_utc':datetime.now(timezone.utc).isoformat(),'input_hashes':hashes,'ids':[r['id'] for r in rows],
        'prototypes':{'static_residual':'captured + temporal mean(restored-captured): preserves captured frame differences before clipping/rounding',
            'dynamic_residual':'captured + restored-captured - temporal mean(restored-captured): keeps restored frame differences before clipping/rounding',
            'half_blend':'0.5*captured+0.5*restored; ordinary blending control'},
        'audio_probe':'Original versus circular +/-0.32 second shifts, same samples and duration. Synthetic mismatch control, not a physical synchronization estimate; wrap boundary and short-context confounds.',
        'risk':'Mean absolute difference between restored and captured response contrasts, where contrast=original-audio AV score minus average shifted-audio AV score. No fitted threshold or inversion after results.',
        'quality_frames':'First, midpoint, last; MUSIQ and BRISQUE. Reference SSIM diagnostic. No human-perception claim.',
        'decision':'A prototype merits independent solution testing only if it has fewer errors than captured-only in aggregate, no source has more errors than captured-only, and both mean MUSIQ and mean BRISQUE improve over captured. All conditions at full coverage. Operational exploratory criterion, not statistical significance.',
        'limits':['No motion compensation; pixel temporal mean is not semantic appearance/motion disentanglement','Clipping can alter nominal temporal preservation','Single detector, no Qwen calls','Prototypes and audio-shift controls are not claimed individually novel','Candidate-harm discrimination is easy when no original restorations correct errors']}
    pp=P/'temporal_evidence_protocol.json';out=P/'results/temporal_evidence_measurements.json'
    if out.exists():
        result=json.loads(out.read_text());old=json.loads(pp.read_text());assert old['input_hashes']==hashes and result['script_sha256']==sha(__file__);protocol=old
    else:
        save(pp,protocol);result=dict(status='RUNNING',script_sha256=sha(__file__),protocol_sha256=sha(pp),cases=[]);save(out,result)
    for folder in ['tools/quality_cpu','tools/iqa_models','tools/python','scripts']:sys.path.insert(0,str(root/folder))
    import cv2,torch,pyiqa
    from skimage.metrics import structural_similarity
    from evaluate_causal import TalkNetCPU
    torch.set_num_threads(2);cv2.setNumThreads(2)
    detector=TalkNetCPU();metric=pyiqa.create_metric('musiq',device='cpu').eval()
    bdir=root/'external/brisque';brisque=cv2.quality.QualityBRISQUE_create(str(bdir/'brisque_model_live.yml'),str(bdir/'brisque_range_live.yml'))
    result['detector_sha256']=detector.checkpoint_sha256
    for row in rows:
        if any(x['id']==row['id'] for x in result['cases']):continue
        c=row['case'];name=row['candidate'];assert sha(c['path'])==c['sha256'] and sha(c['outputs'])==c['output_sha256']
        with np.load(c['path']) as z: audio=z['audio'].copy();labels=z['labels'].copy()
        assert hashlib.sha256(audio.tobytes()).hexdigest()==c['audio_array_sha256']
        with np.load(c['outputs']) as z: captured=z['captured'].copy();restored=z[name].copy();reference=z['reference'].copy()
        videos,clipped=construct(captured,restored);videos={'captured':captured,'restored':restored,**videos}
        T,H,W,_=captured.shape;boxes=[[0,0,W,H]]*T;indices=None;metrics={}
        for variant,video in videos.items():
            pred,idx=detector.predict(video,audio,boxes,25)
            if indices is None:indices=idx
            assert np.array_equal(indices,idx)
            basekey='captured' if variant=='captured' else name if variant=='restored' else None
            if basekey:
                old=c['candidates'][basekey];assert np.allclose(pred['AV'],old['probability'],atol=1e-6)
                musiq,br=old['musiq'],old['brisque'];ssim=old['ssim']
            else:
                iq=[];bq=[]
                for i in [0,(T-1)//2,T-1]:
                    bq.append(float(brisque.compute(cv2.cvtColor(video[i],cv2.COLOR_RGB2BGR))[0]))
                    with torch.inference_mode():iq.append(float(metric(torch.from_numpy(video[i].copy()).permute(2,0,1).unsqueeze(0).float()/255).item()))
                musiq=float(np.mean(iq));br=float(np.mean(bq));ssim=float(np.mean([structural_similarity(a,b,channel_axis=2,data_range=255) for a,b in zip(reference,video)]))
            metrics[variant]=dict(probability=pred['AV'].tolist(),audio_probability=pred['A'].tolist(),musiq=musiq,brisque=br,ssim=ssim,errors=int(np.sum((pred['AV']>=.5)!=labels[idx])),clipped_fraction=clipped.get(variant,0))
        assert all(m['audio_probability']==metrics['captured']['audio_probability'] for m in metrics.values())
        probes={}
        for shift in [-5120,5120]:
            pair={}
            for variant in ['captured','restored']:
                pred,idx=detector.predict(videos[variant],np.roll(audio,shift),boxes,25);assert np.array_equal(idx,indices)
                pair[variant]={k:v.tolist() for k,v in pred.items()}
            assert pair['captured']['A']==pair['restored']['A'];probes[str(shift)]=pair
        contrasts={k:np.asarray(metrics[k]['probability'])-np.mean([probes[s][k]['AV'] for s in probes],axis=0) for k in ['captured','restored']}
        risk=float(np.mean(abs(contrasts['restored']-contrasts['captured'])))
        entry=dict(id=row['id'],source=c['source'],candidate=name,labels=labels[indices].tolist(),indices=indices.tolist(),metrics=metrics,audio_probes=probes,
            contrast={k:v.tolist() for k,v in contrasts.items()},alignment_response_change=risk,input_sha256=c['sha256'],output_sha256=c['output_sha256'],
            original_stats=stats(metrics['captured']['probability'],metrics['restored']['probability'],labels[indices]))
        result['cases'].append(entry);save(out,result)
        print(row['id'],{k:v['errors'] for k,v in metrics.items()},'alignment response change',round(risk,5),flush=True)
    result.update(status='COMPLETE',finished_utc=datetime.now(timezone.utc).isoformat());save(out,result)
    print('Completed all 17 pairs: 153 detector calls; no new Qwen responses.',flush=True)

if __name__=='__main__':main()

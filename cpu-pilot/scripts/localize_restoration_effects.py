"""Exploratory paired spatial intervention; no new agent responses or novelty claim."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)

def masks(height, width):
    # Fixed complementary equal-area masks, feathered over 8 pixels at the midline.
    lower = np.clip((np.arange(height, dtype=float)+.5-height/2)/8+.5, 0, 1)
    lower = np.repeat(lower[:,None], width, axis=1)[None,:,:,None]
    return {'lower':lower, 'upper':1-lower}

def mix(captured, restored, mask):
    return np.rint(captured.astype(float)*mask+restored.astype(float)*(1-mask)).astype(np.uint8)

def stats(c, r, y):
    c,r,y=map(np.asarray,(c,r,y))
    cb,rb=c>=.5,r>=.5
    return dict(captured_errors=int(np.sum(cb!=y)), restored_errors=int(np.sum(rb!=y)),
                harm=int(np.sum((cb==y)&(rb!=y))), corrections=int(np.sum((cb!=y)&(rb==y))), n=len(y))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime-root',type=Path,required=True)
    root=parser.parse_args().runtime_root.resolve()
    rows=[];inputs={}
    for name in ['transfer_rq_measurements','benefit_measurements']:
        file=P/'results'/f'{name}.json';d=json.loads(file.read_text());assert d['status']=='COMPLETE'
        inputs[name]=sha(file)
        for case in d['cases']:
            for candidate in (['fastdvdnet','spatial_nlm'] if name=='transfer_rq_measurements' else ['restored']):
                rows.append(dict(case=case,candidate=candidate,id=case['id']+'__'+candidate))
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(), 'scope':'Exploratory reuse of all 17 cached comparisons, 7 tracks, 5 sources. Prior outcomes were known; not independent confirmation.',
        'input_hashes':inputs,'case_ids':[r['id'] for r in rows],
        'intervention':'Replace lower half of restored face crop with captured pixels; complementary upper-half equal-area control. Fixed horizontal split, 8 pixel feather. Coarse regions, not anatomical mouth landmarks.',
        'primary':'Among captured-correct/restored-wrong timestamps, compare errors rescued by lower versus upper replacement. Report per source, not independent-frame significance.',
        'secondary':'All-case errors, introduced errors from each hybrid, score movement, and fixed temporal residual scores vs MUSIQ, BRISQUE, confidence and disagreement.',
        'confounds':'Hybrid images can be out of distribution; regions have equal area but not equal change energy. Static crop position and head motion confound anatomical interpretation. Audio held fixed.',
        'decision':'Only advance the lower-face mechanism to independent confirmation if lower replacement rescues more harmful timestamps than upper replacement on at least two distinct sources; exploratory operational gate, not statistical significance.',
        'not_tested':['Qwen response to hybrids','anatomically localized mouth causality','new risk estimator','full research novelty','human perceptual preference']}
    pp=P/'locality_protocol.json';dest=P/'results/locality_measurements.json'
    if dest.exists():
        previous=json.loads(dest.read_text()); old=json.loads(pp.read_text())
        assert old['input_hashes']==inputs and previous['script_sha256']==sha(__file__)
        protocol=old;result=previous
    else:
        save(pp,protocol);result=dict(status='RUNNING',script_sha256=sha(__file__),protocol_sha256=sha(pp),cases=[])
        save(dest,result)
    for folder in ['tools/python','scripts']:
        sys.path.insert(0,str(root/folder))
    from evaluate_causal import TalkNetCPU
    detector=TalkNetCPU();result['detector_checkpoint_sha256']=detector.checkpoint_sha256
    for row in rows:
        if any(c['id']==row['id'] for c in result['cases']):continue
        c=row['case'];name=row['candidate']
        assert sha(c['path'])==c['sha256'] and sha(c['outputs'])==c['output_sha256']
        with np.load(c['path']) as b: audio=b['audio'].copy();labels=b['labels'].copy()
        assert hashlib.sha256(audio.tobytes()).hexdigest()==c['audio_array_sha256']
        with np.load(c['outputs']) as b: captured=b['captured'].copy();restored=b[name].copy()
        T,H,W,_=captured.shape;mask=masks(H,W)
        assert np.allclose(mask['lower']+mask['upper'],1) and np.isclose(mask['lower'].sum(),mask['upper'].sum())
        assert np.array_equal(mix(captured,captured,mask['lower']),captured)
        videos={'captured':captured,'restored':restored,**{f'retain_{k}':mix(captured,restored,m) for k,m in mask.items()}}
        measured={};indices=None
        for variant,video in videos.items():
            probs,idx=detector.predict(video,audio,[[0,0,W,H]]*T,25)
            if indices is None:indices=idx
            assert np.array_equal(indices,idx)
            measured[variant]={k:v.tolist() for k,v in probs.items()}
        for variant,key in [('captured','captured'),('restored',name)]:
            expected=np.asarray(c['candidates'][key]['probability'])
            assert np.allclose(measured[variant]['AV'],expected,atol=1e-6), 'Baseline reproduction failed'
        assert all(measured[v]['A']==measured['captured']['A'] for v in measured)
        y=labels[indices];cp=np.array(measured['captured']['AV']);rp=np.array(measured['restored']['AV'])
        harm=(cp>=.5)==y;harm &= ((rp>=.5)!=y)
        residual=(restored.astype(float)-captured.astype(float))/255
        features={}
        for region,m in {**mask,'global':np.ones_like(mask['lower'])}.items():
            features[region+'_spatial_change']=float(np.sum(abs(residual)*m)/(T*3*m.sum()))
            features[region+'_temporal_change']=float(np.sum(abs(np.diff(residual,axis=0))*m)/((T-1)*3*m.sum()))
        hybrid={}
        for region in mask:
            hp=np.array(measured['retain_'+region]['AV']);hb=hp>=.5
            hybrid[region]={'errors':int(np.sum(hb!=y)), 'rescued_restoration_harm':int(np.sum(harm&(hb==y))),
                'new_errors_vs_captured':int(np.sum(((cp>=.5)==y)&(hb!=y))),
                'score_distance_to_captured':float(np.mean(abs(hp-cp))),
                'mean_absolute_pixel_change_from_restored':float(np.mean(abs(videos['retain_'+region].astype(float)-restored.astype(float))))}
        entry={'id':row['id'],'source':c['source'],'entity':c['entity'],'start':c['start'],'candidate':name,
            'input_sha256':c['sha256'],'output_sha256':c['output_sha256'], 'labels':y.tolist(), 'indices':indices.tolist(),
            'stats':stats(cp,rp,y),'hybrids':hybrid,'features':features,'predictions':measured,
            'quality':{k:{m:c['candidates'][k][m] for m in ['musiq','brisque']} for k in ['captured',name]}}
        result['cases'].append(entry);save(dest,result)
        print(row['id'],entry['stats'],'rescued lower/upper',hybrid['lower']['rescued_restoration_harm'],hybrid['upper']['rescued_restoration_harm'],flush=True)
    result.update(status='COMPLETE',finished_utc=datetime.now(timezone.utc).isoformat());save(dest,result)
    print('All 17 comparisons complete; 68 detector inferences, original audio retained.',flush=True)

if __name__=='__main__': main()

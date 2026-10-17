"""Rescore cached real AVA candidates with an explicit, task-matched checkpoint."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from talknet_checkpoint_cpu import load_detector

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime-root',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--measurements',type=Path,nargs='+',required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    inputs=[]
    for p in args.measurements:
        d=json.loads(p.read_text())
        if d['status'] not in ('COMPLETE','COMPLETE_WITH_EXCLUSIONS'):raise ValueError('Input incomplete')
        inputs.extend(d['cases'])
    if len({c['id'] for c in inputs})!=len(inputs):raise ValueError('Duplicate cases')
    detector=load_detector(args.runtime_root,args.checkpoint)
    result=dict(status='RUNNING',scope='AVA training sources only; checkpoint correction and restoration characterization, not final validation',
        code_sha256=sha(Path(__file__)),detector_checkpoint_sha256=detector.checkpoint_sha256,inputs={str(p):sha(p) for p in args.measurements},cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True);save(args.output,result)
    for case in inputs:
        bundle=Path(case['local_bundle'])
        if sha(bundle)!=case['bundle_sha256']:raise ValueError('Bundle changed')
        with np.load(bundle,allow_pickle=False) as z:
            audio=z['audio'].copy();labels=z['labels'].copy();candidates={}
            for name in ('captured','fastdvdnet','spatial_nlm','reference'):
                video=z[name];T,H,W,_=video.shape;prob,indices=detector.predict(video,audio,[[0,0,W,H]]*T,25)
                candidates[name]=dict(probability=prob['AV'].tolist(),musiq=case['candidates'][name]['musiq'],brisque=case['candidates'][name]['brisque'],
                    error_count=int(np.sum((prob['AV']>=.5)!=labels[indices])),n=len(indices))
                print(case['id'],name,candidates[name]['error_count'],'/',len(indices),flush=True)
        entry={k:case[k] for k in ('id','source','category','start','entity','local_bundle','bundle_sha256')}
        entry.update(labels=labels[indices].tolist(),scored_indices=indices.tolist(),candidates=candidates)
        result['cases'].append(entry);save(args.output,result)
    result['status']='COMPLETE';save(args.output,result)
if __name__=='__main__':main()

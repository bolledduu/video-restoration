"""Training-only checkpoint sanity check against the authors' AVA weights."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    for folder in ('tools/quality_cpu','tools/iqa_models','tools/python','scripts'):sys.path.insert(0,str(args.runtime_root/folder))
    import torch
    from evaluate_causal import TalkNetCPU
    detector=TalkNetCPU();old=detector.checkpoint_sha256;checkpoint=args.runtime_root/'external/TalkNet-ASD/pretrain_AVA.model'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    detector.model.load_state_dict({k.removeprefix('model.'):v for k,v in state.items() if k.startswith('model.')},strict=True)
    for name,head in detector.heads.items():
        prefix='loss'+name+'.FC.';head.load_state_dict({k.removeprefix(prefix):v for k,v in state.items() if k.startswith(prefix)},strict=True)
    pilot=Path(__file__).resolve().parents[1];rows=[]
    for directory in ('ava_smoke_v1','ava_corrected_dev_v3','cts_train_JNb_v1'):
        document=json.loads((pilot/'results'/directory/'measurements.json').read_text())
        for case in document['cases']:
            path=Path(case['local_bundle'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=case['bundle_sha256']:raise ValueError('Source bundle changed')
            with np.load(path,allow_pickle=False) as z:video=z['reference'].copy();audio=z['audio'].copy();labels=z['labels'].copy()
            T,H,W,_=video.shape;prob,idx=detector.predict(video,audio,[[0,0,W,H]]*T,25);truth=labels[idx]
            row=dict(id=case['id'],source=case['source'],category=case['category'],indices=idx.tolist(),labels=truth.tolist(),
                ava_checkpoint_probability=prob['AV'].tolist(),ava_checkpoint_errors=int(np.sum((prob['AV']>=.5)!=truth)),
                talkset_checkpoint_errors=case['candidates']['reference']['error_count'],eligible=len(truth),
                false_positive=int(np.sum((prob['AV']>=.5)&(truth==0))),false_negative=int(np.sum((prob['AV']<.5)&(truth==1))))
            rows.append(row);print(row['source'],row['category'],'TalkSet reference errors',row['talkset_checkpoint_errors'],'AVA reference errors',row['ava_checkpoint_errors'],'/',row['eligible'],flush=True)
    with checkpoint.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    args.output.write_text(json.dumps(dict(scope='Training-source checkpoint diagnostic on original source crops; no method selection or final-test inference',talkset_checkpoint_sha256=old,ava_checkpoint_sha256=digest,
        preprocessing='Same cached tight face crops, grayscale 112x112, MFCC features; softmax threshold 0.5. AVA dataLoader uses grayscale 112x112; not the full official benchmark protocol.',rows=rows),indent=2))
if __name__=='__main__':main()

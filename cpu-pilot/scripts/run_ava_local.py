"""Real AVA smoke measurements; preserve source PTS, official labels and provenance."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1048576):h.update(b)
    return h.hexdigest()

def save(p,obj):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(p)

def trim_audio_rounding(audio, expected, tolerance=16):
    """Discard at most 1 ms of excess 16 kHz resampler output; never pad gaps."""
    excess = len(audio) - expected
    if not 0 <= excess <= tolerance:
        raise RuntimeError(f'Audio duration mismatch: {len(audio)} samples, expected {expected}')
    return audio[:expected].copy(), excess

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-sources',type=int,default=1);ap.add_argument('--data-root',type=Path);ap.add_argument('--media-manifest',type=Path);ap.add_argument('--episode-id');args=ap.parse_args()
    root=args.runtime_root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    for f in ['tools/quality_cpu','tools/iqa_models','tools/python','tools/media_runtime','scripts']:sys.path.insert(0,str(root/f))
    import av, cv2, torch, numpy as np, pyiqa, imageio_ffmpeg
    from evaluate_causal import TalkNetCPU
    sys.path.insert(0,str(root/'external/fastdvdnet'))
    from models import FastDVDnet
    from fastdvdnet import denoise_seq_fastdvdnet
    data=args.data_root or root/'data/ava_agent';mp=data/'manifest.json';manifest=json.loads(mp.read_text())
    if args.media_manifest:
        receipts=json.loads(args.media_manifest.read_text())['sources']
        downloads={x['source']:{'path':x['path'],'sha256':sha(x['path'])} for x in receipts}
    else:
        downloads={x['source']:x for x in json.loads((data/'media_download_audit.json').read_text())}
    sources=manifest['selected_sources'][:args.max_sources];episodes=[x for x in manifest['episodes'] if x['source'] in sources]
    if args.episode_id:
        episodes=[e for e in manifest['episodes'] if e['episode_id']==args.episode_id]
        if len(episodes)!=1:raise ValueError('Episode ID must select exactly one frozen episode')
        sources=[episodes[0]['source']]
    protocol={'dataset':'AVA-ActiveSpeaker v1.0 validation','scope':'One-source smoke test' if len(sources)==1 else 'Frozen integration batch','sources':sources,'episode_ids':[e['episode_id'] for e in episodes],'manifest_sha256':sha(mp),'code_sha256':sha(__file__),'noise_sigma':25,'seed':20261017,'fps':25,'seconds':4,'crop_size':128,'candidate_names':['captured','fastdvdnet','spatial_nlm'],'quality_frames':[0,49,99],'label_policy':'Official published annotations; inaudible speech excluded; no independent relabeling','alignment_policy':'Select decoded frames by absolute PTS; nearest official face annotation within 25ms; original audio segment at matching source time; technical alignment only, human sync review pending'}
    protocol['source_sha256']={source:downloads[source]['sha256'] for source in sources}
    pp=out/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise RuntimeError('Configuration mismatch: use a new output directory')
    save(pp,protocol)
    for source in sources:
        receipt=downloads[source];p=Path(receipt['path'])
        if sha(p)!=receipt['sha256']:raise RuntimeError('Source media hash mismatch')
    torch.set_num_threads(2);torch.manual_seed(0);cv2.setNumThreads(2)
    detector=TalkNetCPU();restorer=FastDVDnet().cpu().eval();weight=root/'external/fastdvdnet/model_clipped_noise.pth'
    state=torch.load(weight,map_location='cpu',weights_only=True);restorer.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()})
    metric=pyiqa.create_metric('musiq',device='cpu').eval();bd=root/'external/brisque'
    brisque=cv2.quality.QualityBRISQUE_create(str(bd/'brisque_model_live.yml'),str(bd/'brisque_range_live.yml'))
    weights={'talknet':detector.checkpoint_sha256,'fastdvdnet':sha(weight),'musiq':sha(Path(torch.hub.get_dir())/'pyiqa/musiq_koniq_ckpt-e95806b9.pth'),'brisque_model':sha(bd/'brisque_model_live.yml'),'brisque_range':sha(bd/'brisque_range_live.yml')}
    rp=out/'measurements.json';result={'status':'RUNNING','protocol':protocol,'weights':weights,'expected_episodes':len(episodes),'cases':[],'exclusions':[]}
    if rp.exists():
        result=json.loads(rp.read_text());assert result['protocol']==protocol and result['weights']==weights,'Resume identity differs'
    done={e['id'] for e in result['cases']}|{e['id'] for e in result['exclusions']}
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    for episode in episodes:
        sid=episode['episode_id']
        if sid in done:continue
        start=episode['start'];source=Path(downloads[episode['source']]['path']);annotations=episode['annotations']
        container=av.open(str(source));stream=container.streams.video[0];container.seek(int((start-.15)/stream.time_base),stream=stream,backward=True)
        frames=[];pts=[]
        for frame in container.decode(stream):
            t=float(frame.pts*stream.time_base)
            if t<start-.1:continue
            if t>start+4.05:break
            frames.append(frame.to_ndarray(format='rgb24'));pts.append(t)
        container.close();grid=start+np.arange(100)/25
        if not frames:raise RuntimeError('No decoded frames at source timestamps')
        chosen=np.abs(np.array(pts)[:,None]-grid[None,:]).argmin(axis=0)
        max_delta=float(np.max(np.abs(np.array(pts)[chosen]-grid)))
        if max_delta>.025:raise RuntimeError('Decoded timestamp alignment exceeds tolerance')
        times=np.array([float(r[1]) for r in annotations]);crops=[];labels=[];sizes=[];label_deltas=[]
        for k in chosen:
            frame=frames[k];h,w=frame.shape[:2];j=int(np.argmin(abs(times-pts[k])));row=annotations[j]
            label_deltas.append(abs(times[j]-pts[k]));x1,y1,x2,y2=map(float,row[2:6]);x1,y1,x2,y2=int(x1*w),int(y1*h),int(x2*w),int(y2*h)
            if not 0<=x1<x2<=w or not 0<=y1<y2<=h:raise RuntimeError('Invalid official face box')
            sizes.append(min(x2-x1,y2-y1));crops.append(cv2.resize(frame[y1:y2,x1:x2],(128,128),interpolation=cv2.INTER_CUBIC));labels.append({'SPEAKING_AUDIBLE':1,'NOT_SPEAKING':0}[row[6]])
        if max(label_deltas)>.025:raise RuntimeError('Official-label alignment exceeds tolerance')
        if min(sizes)<32:
            result['exclusions'].append({'id':sid,'reason':'native_face_below_32px','minimum':min(sizes)});save(rp,result);continue
        audio_cmd=[ff,'-v','error','-ss',str(start),'-i',str(source),'-t','4','-map','0:a:0','-ac','1','-ar','16000','-f','f32le','pipe:1']
        audio=np.frombuffer(subprocess.run(audio_cmd,capture_output=True,check=True,timeout=90).stdout,dtype='<f4').copy()
        audio,audio_excess_samples=trim_audio_rounding(audio,64000)
        reference=np.stack(crops);rng=np.random.default_rng(20261017+int(sid[:8],16));captured=np.rint(np.clip(reference.astype(float)+rng.normal(0,25,reference.shape),0,255)).astype(np.uint8)
        x=torch.from_numpy(captured.transpose(0,3,1,2).copy()).float()/255
        with torch.inference_mode():restored=denoise_seq_fastdvdnet(x,torch.tensor([25/255]),5,restorer)
        videos={'captured':captured,'fastdvdnet':(restored.permute(0,2,3,1).numpy().clip(0,1)*255).round().astype(np.uint8),'spatial_nlm':np.stack([cv2.cvtColor(cv2.fastNlMeansDenoisingColored(cv2.cvtColor(f,cv2.COLOR_RGB2BGR),None,10,10,7,21),cv2.COLOR_BGR2RGB) for f in captured])}
        case={'audio_excess_samples_trimmed':audio_excess_samples,'id':sid,'source':episode['source'],'category':episode['category'],'start':start,'entity':episode['entity'],'max_frame_grid_delta':max_delta,'max_label_delta':max(label_deltas),'min_native_face':min(sizes),'audio_sha256':hashlib.sha256(audio.tobytes()).hexdigest(),'candidates':{}}
        audios=[]
        for name,video in {**videos,'reference':reference}.items():
            probs,indices=detector.predict(video,audio,[[0,0,128,128]]*100,25);audios.append(probs['A']);truth=np.array(labels)[indices]
            bs=[];ms=[]
            for k in [0,49,99]:
                bs.append(float(brisque.compute(cv2.cvtColor(video[k],cv2.COLOR_RGB2BGR))[0]))
                with torch.inference_mode():ms.append(float(metric(torch.from_numpy(video[k].copy()).permute(2,0,1).unsqueeze(0).float()/255).item()))
            case['candidates'][name]={'probability':probs['AV'].tolist(),'musiq':float(np.mean(ms)),'brisque':float(np.mean(bs)),'error_count':int(np.sum((probs['AV']>=.5)!=truth)),'n':len(truth)}
            case['labels']=truth.tolist();case['scored_indices']=indices.tolist()
            print(sid,name,case['candidates'][name]['error_count'],'/',len(truth),flush=True)
        assert all(np.array_equal(audios[0],v) for v in audios),'Audio branch changed'
        case['audio_branch_invariant']=True
        bundle=out/(sid+'_smoke_outputs.npz');np.savez_compressed(bundle,reference=reference,audio=audio,labels=np.array(labels),**videos)
        case['local_bundle']=str(bundle);case['bundle_sha256']=sha(bundle)
        sheet=np.concatenate([np.concatenate([reference[i],captured[i],videos['fastdvdnet'][i]],axis=1) for i in [0,49,99]],axis=0)
        cv2.imwrite(str(out/(sid+'_review.jpg')),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
        result['cases'].append(case);save(rp,result)
    result['status']='COMPLETE_WITH_EXCLUSIONS' if result['exclusions'] else 'COMPLETE';save(rp,result)

if __name__=='__main__':main()

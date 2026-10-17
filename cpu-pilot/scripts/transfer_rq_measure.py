"""Fixed balanced source-transfer pilot: select by annotations before inference."""
import argparse, csv, hashlib, io, json, sys, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--runtime-root',type=Path,required=True)
    root=parser.parse_args().runtime_root.resolve(); package=Path(__file__).resolve().parents[1]
    out=package/'data/transfer_rq'; out.mkdir(parents=True,exist_ok=True)
    protocol={'sources':['ABqo634gI5s','qv3-HaaxGUc'],'labels':[0,1],
        'selection':'Earliest valid one-second constant-label track per source and label; ties by entity. No inference-based selection.',
        'sigma':25,'seed':20261016,'candidates':['captured','fastdvdnet','spatial_nlm'],
        'quality_frames':[0,12,24],'model':'qwen/qwen3.8-27b','repetitions':2,
        'primary_rq1':'Repeated new Qwen timestamp errors on jointly answered timestamps while both MUSIQ and BRISQUE improve.',
        'secondary':'Majority answers, single-metric improvement, coverage gains/losses and detector errors.',
        'rq2':'Compare fixed quality, confidence and disagreement risk rankings; AUROC undefined without both harmful and non-harmful candidates. No learned assessor claimed.',
        'rq3':'Frozen deterministic guards vs captured-only and restored-only: errors, coverage, and common-answered comparisons. No novel framework claimed.',
        'rq4':'Source transfer and two restoration families, same Qwen model and one synthetic degradation severity; not full generalization validation.',
        'investment_screen':'Support focused method development only if at least one new source shows repeatable primary harm and at least one track shows repeatable correction of a captured Qwen error. Otherwise do not launch full framework experiments from this pilot.',
        'scope':'New to paired Qwen test, not untouched sources in the broader project. Same dataset. Preserve all selected cases and negative outcomes.'}
    pp=package/'transfer_rq_protocol.json'; pp.write_text(json.dumps(protocol,indent=2)+'\n')
    for folder in ['tools/quality_cpu','tools/iqa_models','tools/python','scripts','external/fastdvdnet']:
        sys.path.insert(0,str(root/folder))
    from PIL import Image
    from scipy.io import wavfile
    records=[]
    base=root/'data/unitalk'
    for source in protocol['sources']:
        annotation=base/(source+'.csv'); groups=defaultdict(list)
        for row in csv.reader(annotation.read_text().splitlines()): groups[row[7]].append(row)
        options={0:[],1:[]}
        for entity,track in groups.items():
            track.sort(key=lambda r:float(r[1])); origin=float(track[0][1])
            for k in range(len(track)-24):
                window=track[k:k+25]; label=int(window[0][8])
                if label in options and all(int(r[8])==label for r in window) and np.allclose(np.diff([float(r[1]) for r in window]),.04,atol=1e-4):
                    options[label].append((float(window[0][1]),entity,origin,window))
        with zipfile.ZipFile(base/(source+'_video.zip')) as z, zipfile.ZipFile(base/(source+'_audio.zip')) as az:
            names={(n.split('/')[-2],round(float(Path(n).stem),4)):n for n in z.namelist() if n.endswith('.jpg')}
            for label in protocol['labels']:
                chosen=None
                for start,entity,origin,window in sorted(options[label],key=lambda r:(r[0],r[1])):
                    frames=[]
                    for row in window:
                        name=names.get((entity,round(float(row[1]),4)))
                        if name is None: break
                        im=Image.open(io.BytesIO(z.read(name))).convert('RGB')
                        if min(im.size)<32: break
                        frames.append(np.asarray(im.resize((128,128),Image.Resampling.BICUBIC)))
                    if len(frames)!=25: continue
                    wavnames=[n for n in az.namelist() if Path(n).stem==entity and n.endswith('.wav')]
                    if len(wavnames)!=1: continue
                    rate,pcm=wavfile.read(io.BytesIO(az.read(wavnames[0])))
                    assert rate==16000 and pcm.dtype==np.int16 and pcm.ndim==1
                    offset=round((start-origin)*rate); audio=pcm[offset:offset+16000].astype(np.float32)/32768
                    if len(audio)!=16000: continue
                    chosen=(start,entity,np.stack(frames),audio);break
                assert chosen is not None, f'No valid track for {source} label {label}; do not replace after inference'
                start,entity,ref,audio=chosen; cid=f'{source}__label{label}__sigma25'
                bundle=out/(cid+'_source.npz');np.savez_compressed(bundle,reference=ref,audio=audio,labels=np.full(25,label,dtype=np.int64))
                records.append({'id':cid,'source':source,'label':label,'start':start,'entity':entity,'path':str(bundle),'sha256':sha(bundle),'csv_sha256':sha(annotation),'audio_array_sha256':hashlib.sha256(audio.tobytes()).hexdigest(),'video_archive_sha256':sha(base/(source+'_video.zip')),'audio_archive_sha256':sha(base/(source+'_audio.zip'))})
                print('Selected',cid,'at',start,entity,flush=True)
    manifest={'records':records,'protocol_sha256':sha(pp)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Selection frozen; loading CPU models',flush=True)
    import cv2, torch, pyiqa
    from evaluate_causal import TalkNetCPU
    from models import FastDVDnet
    from fastdvdnet import denoise_seq_fastdvdnet
    from skimage.metrics import structural_similarity
    torch.set_num_threads(2);torch.manual_seed(0);cv2.setNumThreads(2)
    detector=TalkNetCPU();model=FastDVDnet().cpu().eval()
    weight=root/'external/fastdvdnet/model_clipped_noise.pth'
    model.load_state_dict({k.removeprefix('module.'):v for k,v in torch.load(weight,map_location='cpu',weights_only=True).items()},strict=True)
    musiq=pyiqa.create_metric('musiq',device='cpu').eval()
    bd=root/'external/brisque';brisque=cv2.quality.QualityBRISQUE_create(str(bd/'brisque_model_live.yml'),str(bd/'brisque_range_live.yml'))
    dest=package/'results/transfer_rq_measurements.json'
    result={'status':'RUNNING','protocol':protocol,'protocol_sha256':sha(pp),'script_sha256':sha(__file__),'manifest':manifest,
        'weights':{'fastdvdnet':sha(weight),'talknet':detector.checkpoint_sha256,'musiq':sha(Path(torch.hub.get_dir())/'pyiqa/musiq_koniq_ckpt-e95806b9.pth')},'cases':[]}
    if dest.exists():
        prior=json.loads(dest.read_text());assert prior['script_sha256']==result['script_sha256'] and prior['protocol_sha256']==result['protocol_sha256'];result=prior
    for record in records:
        if any(c['id']==record['id'] for c in result['cases']):continue
        with np.load(record['path']) as a: ref,audio,labels=[a[k].copy() for k in ['reference','audio','labels']]
        seed=protocol['seed']+int(hashlib.sha256(record['id'].encode()).hexdigest()[:8],16)
        captured=np.clip(np.rint(ref.astype(float)+np.random.default_rng(seed).normal(0,25,ref.shape)),0,255).astype(np.uint8)
        videos={'captured':captured,'spatial_nlm':np.stack([cv2.cvtColor(cv2.fastNlMeansDenoisingColored(cv2.cvtColor(x,cv2.COLOR_RGB2BGR),None,15,15,7,21),cv2.COLOR_BGR2RGB) for x in captured])}
        with torch.inference_mode(): y=denoise_seq_fastdvdnet(torch.from_numpy(captured.transpose(0,3,1,2).copy()).float()/255,torch.tensor([25/255]),5,model)
        videos['fastdvdnet']=(y.permute(0,2,3,1).numpy().clip(0,1)*255).round().astype(np.uint8)
        videos['reference']=ref; case={**record,'seed':seed,'candidates':{}}; ap=[]
        for name,video in videos.items():
            probs,indices=detector.predict(video,audio,[[0,0,128,128]]*25,25)
            assert len(indices)==24; ap.append(probs['A']);ms=[];bs=[]
            for i in protocol['quality_frames']:
                bs.append(float(brisque.compute(cv2.cvtColor(video[i],cv2.COLOR_RGB2BGR))[0]))
                with torch.inference_mode():ms.append(float(musiq(torch.from_numpy(video[i].copy()).permute(2,0,1).unsqueeze(0).float()/255).item()))
            case['candidates'][name]={'probability':probs['AV'].tolist(),'audio_probability':probs['A'].tolist(),'visual_probability':probs['V'].tolist(),'musiq':float(np.mean(ms)),'brisque':float(np.mean(bs)),'ssim':float(np.mean([structural_similarity(a,b,channel_axis=2,data_range=255) for a,b in zip(ref,video)])),'wrong_tags':int(np.sum((probs['AV']>=.5)!=labels[indices]))}
            print(record['id'],name,'detector errors',case['candidates'][name]['wrong_tags'],flush=True)
        assert all(np.array_equal(ap[0],a) for a in ap)
        path=out/(record['id']+'_outputs.npz');np.savez_compressed(path,**videos)
        case.update(outputs=str(path),output_sha256=sha(path),audio_invariant=True)
        result['cases'].append(case);dest.write_text(json.dumps(result,indent=2)+'\n')
    result['status']='COMPLETE';dest.write_text(json.dumps(result,indent=2)+'\n');print('Measurement complete',flush=True)

if __name__=='__main__': main()

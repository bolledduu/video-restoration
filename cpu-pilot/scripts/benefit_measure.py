"""Fixed two-second benefit screen; no outcome-based clip or severity search."""
import argparse,csv,hashlib,io,json,sys,zipfile,urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime-root',type=Path,required=True)
    root=parser.parse_args().runtime_root.resolve();p=Path(__file__).resolve().parents[1];out=p/'data/benefit_screen';out.mkdir(parents=True,exist_ok=True)
    protocol={'selection':[{'source':'Pc5iQdNpDX0','kind':'speaking'},{'source':'QRXbRSq8p4s','kind':'listening'},{'source':'ho_oK12hY6M','kind':'transition'}],
        'frames':50,'fps':25,'selection_rule':'Earliest available contiguous track per assigned category, ties by entity; transition has 10..40 speaking frames. No outcome selection.',
        'conditions':{'noise25':'Gaussian sigma25; FastDVDnet supplied sigma25','lowres4':'128 to32 bicubic; captured bicubic128, restored pretrained FSRCNN x4','jpeg10':'JPEG quality10; restored bilateral filter diameter5 color25 space3 (classical baseline, not learned JPEG model)'},
        'seed':20261017,'quality_frames':[0,24,49],'repetitions':2,'model':'qwen/qwen3.8-27b',
        'investment_screen':'At least one source with repeated captured-error corrections, at least one with repeated new errors, and fixed-capture-coverage timestamp oracle reduces error by at least 5 percentage points in BOTH repeats.',
        'scope':'Small purposive CPU pilot; source/category confounded, synthetic degradations, same dataset and Qwen model. No claim of representative prevalence or exhaustive restorers.',
        'oracle':'Evaluation labels only; oracle is not deployable. Fixed captured answered timestamps; choose restored only if it corrects a captured error.',
        'fsrcnn_source':'https://github.com/Saafke/FSRCNN_Tensorflow'}
    pp=p/'benefit_screen_protocol.json';pp.write_text(json.dumps(protocol,indent=2)+'\n')
    for folder in ['tools/quality_cpu','tools/iqa_models','tools/python','scripts','external/fastdvdnet']:sys.path.insert(0,str(root/folder))
    from PIL import Image
    from scipy.io import wavfile
    records=[];base=root/'data/unitalk'
    for spec in protocol['selection']:
        source,kind=spec['source'],spec['kind'];groups=defaultdict(list)
        annotation=base/(source+'.csv')
        for row in csv.reader(annotation.read_text().splitlines()):groups[row[7]].append(row)
        options=[]
        for entity,track in groups.items():
            track.sort(key=lambda r:float(r[1]));origin=float(track[0][1])
            for k in range(len(track)-49):
                window=track[k:k+50];ys=[int(r[8]) for r in window]
                valid=(kind=='speaking' and sum(ys)==50) or (kind=='listening' and sum(ys)==0) or (kind=='transition' and 10<=sum(ys)<=40)
                if valid and all(y in (0,1) for y in ys) and np.allclose(np.diff([float(r[1]) for r in window]),.04,atol=1e-4):options.append((float(window[0][1]),entity,origin,window))
        chosen=None
        with zipfile.ZipFile(base/(source+'_video.zip')) as z,zipfile.ZipFile(base/(source+'_audio.zip')) as az:
            names={(n.split('/')[-2],round(float(Path(n).stem),4)):n for n in z.namelist() if n.endswith('.jpg')}
            for start,entity,origin,window in sorted(options,key=lambda x:(x[0],x[1])):
                frames=[]
                for row in window:
                    name=names.get((entity,round(float(row[1]),4)))
                    if name is None:break
                    im=Image.open(io.BytesIO(z.read(name))).convert('RGB')
                    if min(im.size)<32:break
                    frames.append(np.asarray(im.resize((128,128),Image.Resampling.BICUBIC)))
                if len(frames)!=50:continue
                wavnames=[n for n in az.namelist() if Path(n).stem==entity and n.endswith('.wav')]
                if len(wavnames)!=1:continue
                rate,pcm=wavfile.read(io.BytesIO(az.read(wavnames[0])))
                assert rate==16000 and pcm.dtype==np.int16 and pcm.ndim==1
                offset=round((start-origin)*rate);audio=pcm[offset:offset+32000].astype(np.float32)/32768
                if len(audio)!=32000:continue
                chosen=(start,entity,np.stack(frames),audio,np.array([int(r[8]) for r in window]));break
        assert chosen is not None,f'No eligible clip {source}; do not replace based on outcomes'
        start,entity,ref,audio,labels=chosen;bundle=out/(source+'_source.npz');np.savez_compressed(bundle,reference=ref,audio=audio,labels=labels)
        records.append({'source':source,'kind':kind,'start':start,'entity':entity,'path':str(bundle),'sha256':sha(bundle),'csv_sha256':sha(annotation),'audio_array_sha256':hashlib.sha256(audio.tobytes()).hexdigest(),'video_archive_sha256':sha(base/(source+'_video.zip')),'audio_archive_sha256':sha(base/(source+'_audio.zip'))})
        print('Selected',source,kind,start,entity,'speaking frames',int(labels.sum()),flush=True)
    (out/'manifest.json').write_text(json.dumps({'records':records,'protocol_sha256':sha(pp)},indent=2)+'\n')
    modeldir=out/'weights';modeldir.mkdir(exist_ok=True);srpath=modeldir/'FSRCNN_x4.pb'
    if not srpath.exists():urllib.request.urlretrieve('https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x4.pb',srpath)
    import cv2,torch,pyiqa
    from evaluate_causal import TalkNetCPU
    from models import FastDVDnet
    from fastdvdnet import denoise_seq_fastdvdnet
    from skimage.metrics import structural_similarity
    torch.set_num_threads(2);torch.manual_seed(0);cv2.setNumThreads(2)
    detector=TalkNetCPU();model=FastDVDnet().cpu().eval();w=root/'external/fastdvdnet/model_clipped_noise.pth'
    model.load_state_dict({k.removeprefix('module.'):v for k,v in torch.load(w,map_location='cpu',weights_only=True).items()},strict=True)
    sr=cv2.dnn_superres.DnnSuperResImpl_create();sr.readModel(str(srpath));sr.setModel('fsrcnn',4)
    musiq=pyiqa.create_metric('musiq',device='cpu').eval();bd=root/'external/brisque';brisque=cv2.quality.QualityBRISQUE_create(str(bd/'brisque_model_live.yml'),str(bd/'brisque_range_live.yml'))
    dest=p/'results/benefit_measurements.json';result={'status':'RUNNING','protocol':protocol,'protocol_sha256':sha(pp),'script_sha256':sha(__file__),'manifest':records,'weights':{'fsrcnn':sha(srpath),'fastdvdnet':sha(w),'talknet':detector.checkpoint_sha256},'cases':[]}
    if dest.exists():
        prior=json.loads(dest.read_text());assert prior['script_sha256']==result['script_sha256'] and prior['protocol_sha256']==result['protocol_sha256'];result=prior
    for record in records:
        with np.load(record['path']) as a:ref,audio,labels=[a[k].copy() for k in ['reference','audio','labels']]
        reference_metrics=None
        for condition in protocol['conditions']:
            cid=record['source']+'__'+condition
            if any(c['id']==cid for c in result['cases']):continue
            if condition=='noise25':
                seed=protocol['seed']+int(hashlib.sha256(cid.encode()).hexdigest()[:8],16)
                captured=np.clip(np.rint(ref.astype(float)+np.random.default_rng(seed).normal(0,25,ref.shape)),0,255).astype(np.uint8)
                with torch.inference_mode():y=denoise_seq_fastdvdnet(torch.from_numpy(captured.transpose(0,3,1,2).copy()).float()/255,torch.tensor([25/255]),5,model)
                restored=(y.permute(0,2,3,1).numpy().clip(0,1)*255).round().astype(np.uint8)
            elif condition=='lowres4':
                low=[cv2.resize(x,(32,32),interpolation=cv2.INTER_CUBIC) for x in ref]
                captured=np.stack([cv2.resize(x,(128,128),interpolation=cv2.INTER_CUBIC) for x in low])
                restored=np.stack([cv2.cvtColor(sr.upsample(cv2.cvtColor(x,cv2.COLOR_RGB2BGR)),cv2.COLOR_BGR2RGB) for x in low])
            else:
                captured=[]
                for frame in ref:
                    ok,buffer=cv2.imencode('.jpg',cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,10]);assert ok
                    captured.append(cv2.cvtColor(cv2.imdecode(buffer,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB))
                captured=np.stack(captured);restored=np.stack([cv2.bilateralFilter(x,5,25,3) for x in captured])
            videos={'captured':captured,'restored':restored,'reference':ref};case={**record,'id':cid,'condition':condition,'candidates':{}};ap=[]
            for name,video in videos.items():
                if name=='reference' and reference_metrics is not None:
                    case['candidates'][name]=reference_metrics;continue
                probs,indices=detector.predict(video,audio,[[0,0,128,128]]*50,25);assert len(indices)==49
                ap.append(probs['A']);ms=[];bs=[]
                for i in protocol['quality_frames']:
                    bs.append(float(brisque.compute(cv2.cvtColor(video[i],cv2.COLOR_RGB2BGR))[0]))
                    with torch.inference_mode():ms.append(float(musiq(torch.from_numpy(video[i].copy()).permute(2,0,1).unsqueeze(0).float()/255).item()))
                case['candidates'][name]={'probability':probs['AV'].tolist(),'audio_probability':probs['A'].tolist(),'musiq':float(np.mean(ms)),'brisque':float(np.mean(bs)),'ssim':float(np.mean([structural_similarity(a,b,channel_axis=2,data_range=255) for a,b in zip(ref,video)])),'wrong_tags':int(np.sum((probs['AV']>=.5)!=labels[indices]))}
                if name=='reference':reference_metrics=case['candidates'][name]
                print(cid,name,'detector errors',case['candidates'][name]['wrong_tags'],flush=True)
            assert all(np.array_equal(ap[0],v) for v in ap)
            path=out/(cid+'_outputs.npz');np.savez_compressed(path,**videos)
            case.update(outputs=str(path),output_sha256=sha(path),indices=indices.tolist(),scored_labels=labels[indices].tolist())
            result['cases'].append(case);dest.write_text(json.dumps(result,indent=2)+'\n')
    result['status']='COMPLETE';dest.write_text(json.dumps(result,indent=2)+'\n');print('Benefit measurements complete',flush=True)

if __name__=='__main__':main()

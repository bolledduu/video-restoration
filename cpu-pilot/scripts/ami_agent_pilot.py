"""Small separate AMI meeting-domain check with published transcript proxy labels."""
import json, math, sys, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
import av, numpy as np
from scipy.io import wavfile
from PIL import Image, ImageDraw
from quick_temporal_pilot import ROOT, sha, cv2, structural_similarity

def prepare():
    out=ROOT/'data/agent_pilot/ami';pp=ROOT/'ami_agent_protocol.json';p=json.loads(pp.read_text())
    with zipfile.ZipFile(out/'annotations.zip') as z:
        root=ET.fromstring(z.read('corpusResources/meetings.xml'))
        meeting=next(x for x in root.iter('meeting') if x.get('observation')=='ES2002a')
        assert next(x for x in meeting if x.get('nxt_agent')=='A').get('camera')=='Closeup1'
        segments={s:[(float(x.get('transcriber_start')),float(x.get('transcriber_end'))) for x in ET.fromstring(z.read(f'segments/ES2002a.{s}.segments.xml'))] for s in 'ABCD'}
        words={s:[(float(x.get('starttime')),float(x.get('endtime')),x.text or '') for x in ET.fromstring(z.read(f'words/ES2002a.{s}.words.xml')) if x.tag=='w' and x.get('starttime') and float(x.get('endtime'))>float(x.get('starttime'))] for s in 'ABCD'}
    positive=next((a,b,'A',1) for a,b in segments['A'] if b-a>=1.4)
    negatives=sorted((a,b,s,0) for s in 'BCD' for a,b in segments[s] if b-a>=1.4 and not any(x<b and y>a for x,y in segments['A']))
    selections=[positive,negatives[0]];records=[]
    rate,pcm=wavfile.read(out/'mix.wav');assert rate==16000 and pcm.ndim==1 and pcm.dtype==np.int16
    haar=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    canvas=Image.new('RGB',(512,160),'white');draw=ImageDraw.Draw(canvas)
    for j,(a,b,s,label) in enumerate(selections):
        start=math.ceil((a+.2)*25)/25;frames=[];times=[]
        with av.open(str(out/'closeup1.avi')) as c:
            for f in c.decode(video=0):
                if f.time<start-1e-6:continue
                if len(frames)==25:break
                frames.append(f.to_ndarray(format='rgb24'));times.append(float(f.time))
        assert len(frames)==25 and np.allclose(times,start+np.arange(25)/25,atol=1e-5)
        x1,y1,x2,y2=130,100,270,245
        ref=np.stack([cv2.resize(f[y1:y2,x1:x2],(128,128)) for f in frames])
        audio=pcm[round(start*rate):round(start*rate)+16000].astype(np.float32)/32768
        assert len(audio)==16000
        sid=f'ES2002a__targetA__label{label}';path=out/(sid+'.npz')
        np.savez_compressed(path,reference=ref,audio=audio,labels=np.full(25,label,dtype=np.int64))
        quote=' '.join(t for x,y,t in words[s] if x<start+1 and y>start)
        records.append(dict(id=sid,source='AMI_ES2002a',label=label,start=start,segment=[a,b],transcript_speaker=s,transcript_excerpt=quote,timestamps=times,box=[x1,y1,x2,y2],path=str(path),sha256=sha(path)))
        canvas.paste(Image.fromarray(ref[12]),(j*256,25));draw.text((j*256,3),f'label {label}, {start:.2f}s',fill='black')
    canvas.save(out/'crops.png')
    manifest=dict(protocol=p,protocol_sha256=sha(pp),records=records,source_hashes={f:sha(out/f) for f in ('annotations.zip','closeup1.avi','mix.wav')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

def main():
    manifest=prepare();print('AMI inputs prepared',flush=True)
    if '--prepare-only' in sys.argv:return
    sys.path.insert(0,str(ROOT/'tools/iqa_models'))
    import torch,pyiqa
    from evaluate_causal import TalkNetCPU
    sys.path.insert(0,str(ROOT/'external/fastdvdnet'))
    from models import FastDVDnet
    from fastdvdnet import denoise_seq_fastdvdnet
    torch.set_num_threads(2);cv2.setNumThreads(2)
    model=FastDVDnet().cpu().eval();weight=ROOT/'external/fastdvdnet/model_clipped_noise.pth'
    state=torch.load(weight,map_location='cpu',weights_only=True);model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()})
    detector=TalkNetCPU();musiq=pyiqa.create_metric('musiq',device='cpu').eval()
    bd=ROOT/'external/brisque';brisque=cv2.quality.QualityBRISQUE_create(str(bd/'brisque_model_live.yml'),str(bd/'brisque_range_live.yml'))
    result=dict(manifest=manifest,script_sha256=sha(__file__),fastdvdnet_sha256=sha(weight),talknet_sha256=detector.checkpoint_sha256,cases=[])
    for row in manifest['records']:
        a=np.load(row['path']);ref=a['reference'];audio=a['audio'];labels=a['labels']
        rng=np.random.default_rng(20261015+row['label']);captured=np.clip(np.rint(ref.astype(float)+rng.normal(0,40,ref.shape)),0,255).astype(np.uint8)
        with torch.inference_mode():y=denoise_seq_fastdvdnet(torch.from_numpy(captured.transpose(0,3,1,2).copy()).float()/255,torch.tensor([40/255]),5,model)
        restored=(y.permute(0,2,3,1).numpy().clip(0,1)*255).round().astype(np.uint8)
        case=dict(id=row['id'],label=row['label'],candidates={});audio_scores=[]
        for name,video in [('captured',captured),('fastdvdnet',restored),('reference',ref)]:
            probs,indices=detector.predict(video,audio,[[0,0,128,128]]*25,25);audio_scores.append(probs['A']);bs=[];ms=[]
            for i in (0,12,24):
                bs.append(float(brisque.compute(cv2.cvtColor(video[i],cv2.COLOR_RGB2BGR))[0]))
                with torch.inference_mode():ms.append(float(musiq(torch.from_numpy(video[i].copy()).permute(2,0,1).unsqueeze(0).float()/255).item()))
            case['candidates'][name]=dict(error=float(np.mean((probs['AV']>=.5)!=labels[indices])),probability=probs['AV'].tolist(),brisque=float(np.mean(bs)),musiq=float(np.mean(ms)),ssim=float(np.mean([structural_similarity(u,v,channel_axis=2,data_range=255) for u,v in zip(ref,video)])))
            print(row['id'],name,case['candidates'][name]['error'],flush=True)
        assert all(np.array_equal(audio_scores[0],x) for x in audio_scores)
        case['audio_invariance']=True;result['cases'].append(case)
        np.savez_compressed(Path(row['path']).with_name(row['id']+'_outputs.npz'),captured=captured,fastdvdnet=restored,reference=ref)
        (ROOT/'results/ami_agent_pilot.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()

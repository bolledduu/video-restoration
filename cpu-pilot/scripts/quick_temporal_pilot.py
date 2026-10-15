"""Small new-source temporal-denoising transfer probe; protocol fixed before inference."""
import csv, io, json, sys, zipfile, hashlib, time
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/quality_cpu'))
import cv2
import numpy as np
from PIL import Image
from scipy.io import wavfile
from skimage.metrics import structural_similarity
from prepare_unitalk_cpu import fetch

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(p, out):
    base = ROOT/'data/unitalk'
    old = json.loads((base/'manifest.json').read_text())
    used = {r['source'] for r in old['records']}
    protocol = old['protocol']
    url = f"https://huggingface.co/datasets/{protocol['dataset']}/resolve/{protocol['revision']}/"
    rows, audit = [], []
    for archive in sorted(base.glob('*_video.zip'), key=lambda x:(x.stat().st_size,x.name)):
        source = archive.name.removesuffix('_video.zip')
        if source in used: continue
        groups = defaultdict(list)
        for row in csv.reader((base/(source+'.csv')).read_text().splitlines()): groups[row[7]].append(row)
        options=[]
        for entity, track in groups.items():
            track.sort(key=lambda r:float(r[1]))
            for k in range(len(track)-24):
                window=track[k:k+25]
                if len({r[8] for r in window})==1 and int(window[0][8]) in (0,1) and np.allclose(np.diff([float(r[1]) for r in window]),.04,atol=1e-4):
                    options.append((float(window[0][1]),entity,window,float(track[0][1])))
        chosen=None
        with zipfile.ZipFile(archive) as z:
            names={(n.split('/')[-2],round(float(Path(n).stem),4)):n for n in z.namelist() if n.endswith('.jpg')}
            for start,entity,window,origin in sorted(options,key=lambda x:(x[0],x[1])):
                imgs=[]
                for r in window:
                    n=names.get((entity,round(float(r[1]),4)))
                    if not n: break
                    im=Image.open(io.BytesIO(z.read(n))).convert('RGB')
                    if min(im.size)<32: break
                    imgs.append(np.asarray(im.resize((128,128),Image.Resampling.BICUBIC)))
                if len(imgs)==25:
                    chosen=(start,entity,origin,int(window[0][8]),np.stack(imgs));break
        if chosen is None:
            audit.append({'source':source,'status':'no_eligible_window'});continue
        start,entity,origin,label,ref=chosen
        audiozip=fetch(url+f'clips_audios/val/{source}.zip',base/(source+'_audio.zip'))
        with zipfile.ZipFile(audiozip) as z:
            n=next(n for n in z.namelist() if Path(n).stem==entity and n.endswith('.wav'))
            rate,pcm=wavfile.read(io.BytesIO(z.read(n)))
        assert rate==16000 and pcm.dtype==np.int16 and pcm.ndim==1
        offset=round((start-origin)*rate)
        audio=pcm[offset:offset+16000].astype(np.float32)/32768
        assert len(audio)==16000
        path=out/(source+'.npz')
        np.savez_compressed(path,reference=ref,audio=audio,labels=np.full(25,label,dtype=np.int64))
        rows.append(dict(source=source,entity=entity,start=start,label=label,path=str(path),sha256=sha(path),video_sha256=sha(archive),audio_sha256=sha(audiozip),csv_sha256=sha(base/(source+'.csv'))))
        print('Prepared',source,label,flush=True)
        if len(rows)==3: break
    assert len(rows)==3, 'Not enough eligible new cached sources'
    result=dict(records=rows,audit=audit,excluded_sources=sorted(used),dataset=protocol['dataset'],revision=protocol['revision'])
    (out/'manifest.json').write_text(json.dumps(result,indent=2))
    return result

def main():
    out=ROOT/'data/quick_temporal';out.mkdir(exist_ok=True)
    pp=ROOT/'quick_temporal_protocol.json';p=json.loads(pp.read_text())
    manifest=prepare(p,out)
    # Selection is saved before any inference. Re-running is deterministic.
    sys.path.insert(0,str(ROOT/'tools/iqa_models'))
    import torch, pyiqa
    from evaluate_causal import TalkNetCPU
    torch.set_num_threads(2);torch.manual_seed(0);cv2.setNumThreads(2)
    detector=TalkNetCPU()
    musiq=pyiqa.create_metric('musiq',device='cpu').eval()
    bdir=ROOT/'external/brisque'
    brisque=cv2.quality.QualityBRISQUE_create(str(bdir/'brisque_model_live.yml'),str(bdir/'brisque_range_live.yml'))
    result=dict(protocol=p,protocol_sha256=sha(pp),script_sha256=sha(__file__),manifest=manifest,
        talknet_sha256=detector.checkpoint_sha256,opencv=cv2.__version__,cases=[])
    result['quality_hashes']={n:sha(bdir/n) for n in ('brisque_model_live.yml','brisque_range_live.yml')}
    result['quality_hashes']['musiq']=sha(Path(torch.hub.get_dir())/'pyiqa/musiq_koniq_ckpt-e95806b9.pth')
    rng=np.random.default_rng(p['seed'])
    for row in manifest['records']:
        a=np.load(row['path']);ref=a['reference'];audio=a['audio'];labels=a['labels']
        degraded=np.clip(np.rint(ref.astype(float)+rng.normal(0,p['noise_sigma'],ref.shape)),0,255).astype(np.uint8)
        bgr=[cv2.cvtColor(x,cv2.COLOR_RGB2BGR) for x in degraded]
        pad=[bgr[i] for i in np.pad(np.arange(25),(2,2),mode='reflect')]
        videos={'degraded':degraded,
          'spatial_nlm':np.stack([cv2.cvtColor(cv2.fastNlMeansDenoisingColored(x,None,15,15,7,21),cv2.COLOR_BGR2RGB) for x in bgr]),
          'temporal_nlm':np.stack([cv2.cvtColor(cv2.fastNlMeansDenoisingColoredMulti(pad,i+2,5,None,15,15,7,21),cv2.COLOR_BGR2RGB) for i in range(25)])}
        case=dict(source=row['source'],label=row['label'],candidates={})
        audio_probs=[]
        for name,video in {**videos,'reference':ref}.items():
            probs,indices=detector.predict(video,audio,[[0,0,128,128]]*25,25)
            assert len(indices)==24
            audio_probs.append(probs['A'])
            bs=[];ms=[]
            for i in (0,12,24):
                bs.append(float(brisque.compute(cv2.cvtColor(video[i],cv2.COLOR_RGB2BGR))[0]))
                with torch.inference_mode(): ms.append(float(musiq(torch.from_numpy(video[i].copy()).permute(2,0,1).unsqueeze(0).float()/255).item()))
            case['candidates'][name]=dict(error=float(np.mean((probs['AV']>=.5)!=labels[indices])),probability=probs['AV'].tolist(),
                visual_probability=probs['V'].tolist(),confidence=float(np.mean(np.maximum(probs['AV'],1-probs['AV']))),
                brisque=float(np.mean(bs)),musiq=float(np.mean(ms)),ssim=float(np.mean([structural_similarity(x,y,channel_axis=2,data_range=255) for x,y in zip(ref,video)])))
            print(row['source'],name,case['candidates'][name]['error'],flush=True)
        assert all(np.array_equal(audio_probs[0],x) for x in audio_probs)
        case['audio_invariance']=True
        choices=list(videos)
        c=case['candidates']
        case['winners']={metric:min(choices,key=lambda k:(sign*c[k][metric],k)) for metric,sign in [('brisque',1),('musiq',-1),('confidence',-1),('error',1)]}
        winner=case['winners']['musiq']
        disagreement=float(np.mean((np.array(c[winner]['probability'])>=.5)!=(np.array(c['degraded']['probability'])>=.5)))
        case['disagreement']=disagreement
        case['winners']['guard']='degraded' if disagreement>.1 else winner
        case['winners']['retain']='degraded'
        np.savez_compressed(out/(row['source']+'_outputs.npz'),**videos,reference=ref)
        case['output_sha256']=sha(out/(row['source']+'_outputs.npz'))
        result['cases'].append(case)
        (ROOT/'results/quick_temporal_pilot.json').write_text(json.dumps(result,indent=2))
    rows=result['cases']
    harmed=[r['source'] for r in rows if r['winners']['brisque']==r['winners']['musiq'] and r['winners']['musiq']!='degraded' and r['candidates'][r['winners']['musiq']]['error']-r['candidates']['degraded']['error']>=1/24-1e-9]
    helped=[r['source'] for r in rows if min(r['candidates'][k]['error'] for k in ('spatial_nlm','temporal_nlm'))<r['candidates']['degraded']['error']]
    result['summary']=dict(agreed_harmed_sources=harmed,restoration_helped_sources=helped,feasibility_screen_pass=len(harmed)>=2 and len(helped)>=1,
        mean_error={rule:float(np.mean([r['candidates'][r['winners'][rule]]['error'] for r in rows])) for rule in ('retain','brisque','musiq','confidence','guard','error')})
    (ROOT/'results/quick_temporal_pilot.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['summary'],indent=2),flush=True)

if __name__=='__main__': main()

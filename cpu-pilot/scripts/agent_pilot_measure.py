"""Real CPU measurements; reference/label evaluation stays outside agent tools."""
import json, sys, time
from pathlib import Path
import numpy as np
from quick_temporal_pilot import ROOT, sha, cv2, structural_similarity
sys.path.insert(0,str(ROOT/'tools/iqa_models'))
import torch, pyiqa
from evaluate_causal import TalkNetCPU
sys.path.insert(0,str(ROOT/'external/fastdvdnet'))
from models import FastDVDnet
from fastdvdnet import denoise_seq_fastdvdnet

def main():
    ppath=ROOT/'agent_pilot_protocol.json';p=json.loads(ppath.read_text())
    out=ROOT/'data/agent_pilot';manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['protocol_sha256']==sha(ppath)
    torch.set_num_threads(2);torch.manual_seed(0);cv2.setNumThreads(2)
    detector=TalkNetCPU();model=FastDVDnet().cpu().eval()
    weight=ROOT/'external/fastdvdnet/model_clipped_noise.pth'
    state=torch.load(weight,map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    musiq=pyiqa.create_metric('musiq',device='cpu').eval()
    bd=ROOT/'external/brisque';brisque=cv2.quality.QualityBRISQUE_create(str(bd/'brisque_model_live.yml'),str(bd/'brisque_range_live.yml'))
    dest=ROOT/'results/agent_pilot_measurements.json'
    result=dict(protocol=p,protocol_sha256=sha(ppath),script_sha256=sha(__file__),manifest=manifest,
        weights={'fastdvdnet':sha(weight),'talknet':detector.checkpoint_sha256,'musiq':sha(Path(torch.hub.get_dir())/'pyiqa/musiq_koniq_ckpt-e95806b9.pth')},cases=[])
    if dest.exists():
        old=json.loads(dest.read_text())
        assert old['protocol_sha256']==result['protocol_sha256'] and old['script_sha256']==result['script_sha256'] and old['weights']==result['weights']
        result=old
    done={r['id'] for r in result['cases']}
    for row in manifest['records']:
        assert sha(row['path'])==row['sha256']
        a=np.load(row['path']);ref=a['reference'];audio=a['audio'];labels=a['labels']
        for sigma in p['sigma']:
            sid=row['id']+f'__sigma{sigma}'
            if sid in done:continue
            seed=p['seed']+int(sha(row['path'])[:8],16)+sigma
            rng=np.random.default_rng(seed)
            captured=np.clip(np.rint(ref.astype(float)+rng.normal(0,sigma,ref.shape)),0,255).astype(np.uint8)
            videos={'captured':captured,'spatial_nlm':np.stack([cv2.cvtColor(cv2.fastNlMeansDenoisingColored(cv2.cvtColor(x,cv2.COLOR_RGB2BGR),None,15,15,7,21),cv2.COLOR_BGR2RGB) for x in captured])}
            x=torch.from_numpy(captured.transpose(0,3,1,2).copy()).float()/255
            for name,noise in [('fastdvdnet',sigma),('fastdvdnet_55',55)]:
                with torch.inference_mode():y=denoise_seq_fastdvdnet(x,torch.tensor([noise/255]),5,model)
                videos[name]=(y.permute(0,2,3,1).numpy().clip(0,1)*255).round().astype(np.uint8)
            case=dict(id=sid,source=row['source'],label=row['label'],sigma=sigma,seed=seed,candidates={})
            ap=[]
            for name,video in {**videos,'reference':ref}.items():
                probs,indices=detector.predict(video,audio,[[0,0,128,128]]*25,25)
                assert len(indices)==24
                ap.append(probs['A']);bs=[];ms=[]
                for i in p['quality_frames']:
                    bs.append(float(brisque.compute(cv2.cvtColor(video[i],cv2.COLOR_RGB2BGR))[0]))
                    with torch.inference_mode():ms.append(float(musiq(torch.from_numpy(video[i].copy()).permute(2,0,1).unsqueeze(0).float()/255).item()))
                case['candidates'][name]=dict(error=float(np.mean((probs['AV']>=.5)!=labels[indices])),probability=probs['AV'].tolist(),visual_probability=probs['V'].tolist(),audio_probability=probs['A'].tolist(),confidence=float(np.mean(np.maximum(probs['AV'],1-probs['AV']))),brisque=float(np.mean(bs)),musiq=float(np.mean(ms)),ssim=float(np.mean([structural_similarity(u,v,channel_axis=2,data_range=255) for u,v in zip(ref,video)])),input_mse=float(np.mean((video.astype(float)-captured.astype(float))**2)))
                print(sid,name,'error',case['candidates'][name]['error'],flush=True)
            assert all(np.array_equal(ap[0],v) for v in ap)
            case['audio_invariance']=True
            c=case['candidates'];choices=list(videos)
            case['winners']={m:min(choices,key=lambda n:(s*c[n][m],n)) for m,s in [('brisque',1),('musiq',-1),('confidence',-1),('error',1)]}
            w=case['winners']['musiq'];d=float(np.mean((np.array(c[w]['probability'])>=.5)!=(np.array(c['captured']['probability'])>=.5)))
            case['winners']['guard']='captured' if d>.1 else w
            case['winners']['retain']='captured';case['disagreement']=d
            path=out/(sid+'_outputs.npz');np.savez_compressed(path,**videos,reference=ref)
            case['outputs']=str(path);case['output_sha256']=sha(path)
            result['cases'].append(case);dest.write_text(json.dumps(result,indent=2))
    rows=result['cases']
    harm=sorted({r['source'] for r in rows if r['winners']['brisque']==r['winners']['musiq'] and r['candidates'][r['winners']['musiq']]['error']>=r['candidates']['captured']['error']+1/24-1e-9})
    gain=sorted({r['source'] for r in rows if min(r['candidates'][k]['error'] for k in p['candidates'][1:])<r['candidates']['captured']['error']})
    result['summary']=dict(agreed_harmed_sources=harm,restoration_helped_sources=gain,screen_pass=len(harm)>=2 and len(gain)>=1,mean_error={k:float(np.mean([r['candidates'][r['winners'][k]]['error'] for r in rows])) for k in ('retain','brisque','musiq','confidence','guard','error')})
    dest.write_text(json.dumps(result,indent=2));print(json.dumps(result['summary'],indent=2),flush=True)

if __name__=='__main__':main()

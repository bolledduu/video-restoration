"""CPU TalkNet diagnostic on provisional face crops, not verified accuracy.

Uses official model and classifier weights without CUDA or training optimizer.
Largest-face proposals are shared across conditions; identity tracking is NOT
validated, so predictions must not be scored as benchmark question answers.
"""
import hashlib
import json
import subprocess
import sys
import time
import av
import cv2
import numpy as np
import torch
from scipy.signal import resample_poly
from cpu_media import ROOT

sys.path.insert(0,str(ROOT/'tools'/'python'))
sys.path.insert(0,str(ROOT/'external'/'TalkNet-ASD'))
import python_speech_features
from model.talkNetModel import talkNetModel

def audio_segment(path,start,duration):
    chunks=[];expected_sample=0
    with av.open(str(path)) as c:
        s=c.streams.audio[0];rate=s.codec_context.sample_rate
        for f in c.decode(audio=0):
            position=round(float(f.pts*f.time_base)*rate)
            if abs(position-expected_sample)>1:
                raise ValueError('Audio gap/offset: explicit alignment required')
            x=f.to_ndarray()
            if not f.format.is_planar:x=x.reshape(-1,len(f.layout.channels)).T
            chunks.append(x.mean(axis=0))
            expected_sample+=f.samples
    x=np.concatenate(chunks)
    x=x[round(start*rate):round((start+duration)*rate)]
    from math import gcd
    g=gcd(rate,16000)
    return resample_poly(x,16000//g,rate//g)

def main():
    torch.set_num_threads(2);torch.manual_seed(0);cv2.setNumThreads(2)
    model=talkNetModel().cpu().eval()
    checkpoint=ROOT/'external'/'TalkNet-ASD'/'pretrain_TalkSet.model'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in state.items() if k.startswith('model.')},strict=True)
    heads={}
    for name,size in [('AV',256),('A',128),('V',128)]:
        head=torch.nn.Linear(size,2).eval()
        head.load_state_dict({k.removeprefix('loss'+name+'.FC.'):v for k,v in state.items() if k.startswith('loss'+name+'.FC.')},strict=True)
        heads[name]=head
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    records=[]
    for file in sorted((ROOT/'data'/'baseline').glob('*.npz')):
        arrays=np.load(file);times=arrays['timestamps'];start=float(times[0])
        sample,level=file.stem.split('__')
        # Normalize the complete excerpt to 25 Hz for TalkNet's 4:1 audio/video features.
        n=int(np.floor((times[-1]-times[0]+np.median(np.diff(times)))*25))
        grid=start+np.arange(n)/25
        indices=np.abs(times[:,None]-grid[None,:]).argmin(axis=0)
        boxes=[];usable=True
        for image in arrays['reference'][indices]:
            large=cv2.resize(image,(512,288))
            found=detector.detectMultiScale(cv2.cvtColor(large,cv2.COLOR_RGB2GRAY),1.1,3,minSize=(24,24))
            if len(found):
                x,y,w,h=max(found,key=lambda b:b[2]*b[3]);cx,cy=(x+w/2)/2,(y+h/2)/2
                side=max(w,h)/2*1.6
                box=[max(0,int(cx-side/2)),max(0,int(cy-side/2)),min(256,int(cx+side/2)),min(144,int(cy+side/2))]
            elif boxes:box=boxes[-1]
            else:usable=False;break
            boxes.append(box)
        if not usable:
            records.append({'sample_id':sample,'level':level,'status':'NO_INITIAL_FACE'});continue
        audio=audio_segment(ROOT/'data'/'raw'/(sample+'.mp4'),start,n/25)
        # Match official WAV int16 amplitude convention.
        pcm=np.clip(audio*32768,-32768,32767).astype(np.int16)
        features=python_speech_features.mfcc(pcm,16000,numcep=13,winlen=.025,winstep=.010)
        n=min(n,len(features)//4);features=features[:4*n];indices=indices[:n];boxes=boxes[:n]
        ta=torch.from_numpy(features.copy()).float().unsqueeze(0)
        for condition in ['reference','degraded','restored']:
            crops=[]
            for im,(x1,y1,x2,y2) in zip(arrays[condition][indices],boxes):
                gray=cv2.cvtColor(im[y1:y2,x1:x2],cv2.COLOR_RGB2GRAY)
                crops.append(cv2.resize(gray,(112,112)))
            tv=torch.from_numpy(np.stack(crops)).float().unsqueeze(0)
            began=time.perf_counter()
            with torch.inference_mode():
                ve=model.forward_visual_frontend(tv)
                ae=model.forward_audio_frontend(ta)
                # Pure modality heads BEFORE cross-attention prevent audio leakage into visual-only evaluation.
                logits_v=heads['V'](model.forward_visual_backend(ve)).numpy()
                logits_a=heads['A'](model.forward_audio_backend(ae)).numpy()
                ca,cv=model.forward_cross_attention(ae,ve)
                logits_av=heads['AV'](model.forward_audio_visual_backend(ca,cv)).numpy()
            record={'sample_id':sample,'level':level,'condition':condition,'model':'TalkNet-TalkSet',
                    'device':'cpu','status':'PROVISIONAL_CROP_DIAGNOSTIC','ground_truth_verified':False,
                    'accuracy':None,'false_attribution_rate':None,'frame_count':n,
                    'timestamps':grid[:n].tolist(),'crop_boxes_256x144':boxes,
                    'crop_policy':'largest frontal Haar face per reference frame; previous box on misses; identities NOT verified',
                    'audio_sha256':hashlib.sha256(pcm.tobytes()).hexdigest(),
                    'audio_source':'original segment, fixed across conditions',
                    'logits_visual_only':logits_v.tolist(),'logits_audio_only':logits_a.tolist(),'logits_fixed_audio_visual':logits_av.tolist(),
                    'score_interpretation':'raw two-class logits; uncalibrated, not confidence probabilities',
                    'wall_seconds':time.perf_counter()-began,
                    'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    'code_commit':subprocess.check_output(['git','-C',str(ROOT/'external'/'TalkNet-ASD'),'rev-parse','HEAD'],text=True).strip()}
            records.append(record);print(sample,level,condition,'frames',n,'seconds',round(record['wall_seconds'],2),flush=True)
            (ROOT/'results'/'talknet_diagnostics.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()

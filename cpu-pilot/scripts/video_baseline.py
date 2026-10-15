"""Run official FastDVDnet weights on short real-video excerpts on CPU.

This denoising baseline is task-mismatched to blur/compression and is not a
replacement for an audio-conditioned model or its matched video-only ablation.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import av
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
from cpu_media import ROOT

EXTERNAL=ROOT/'external'/'fastdvdnet'
sys.path.insert(0,str(EXTERNAL))
from models import FastDVDnet
from fastdvdnet import denoise_seq_fastdvdnet

def frames(path, start, duration=2.0):
    images=[];times=[]
    with av.open(str(path)) as c:
        rate=c.streams.video[0].average_rate
        for f in c.decode(video=0):
            if float(f.time)<start:continue
            if float(f.time)>=start+duration:break
            images.append(cv2.resize(f.to_ndarray(format='rgb24'),(256,144),interpolation=cv2.INTER_AREA))
            times.append(float(f.time))
    return np.stack(images),times,rate

def measure(a,b):
    mse=np.mean((a.astype(float)-b.astype(float))**2)
    return {'psnr_db':float(10*np.log10(255**2/mse)) if mse else None,
            'ssim_rgb_mean':float(np.mean([structural_similarity(x,y,channel_axis=2,data_range=255) for x,y in zip(a,b)]))}

def save_video(path,x,rate):
    with av.open(str(path),'w') as c:
        s=c.add_stream('libx264',rate=rate);s.width=256;s.height=144;s.pix_fmt='yuv420p';s.options={'crf':'0'}
        for im in x:
            for p in s.encode(av.VideoFrame.from_ndarray(im,format='rgb24')):c.mux(p)
        for p in s.encode():c.mux(p)

def main():
    torch.set_num_threads(2);torch.manual_seed(0);cv2.setNumThreads(2)
    model=FastDVDnet().cpu().eval();weight=EXTERNAL/'model.pth'
    state=torch.load(weight,map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    dest=ROOT/'data'/'baseline';dest.mkdir(exist_ok=True)
    records=[]
    for source,start in zip(sorted((ROOT/'data'/'raw').glob('*.mp4')),[1.0,6.0,6.5]):
        ref,times,rate=frames(source,start)
        for level in ('moderate','severe'):
            inp,other_times,_=frames(ROOT/'data'/'processed'/(source.stem+'__'+level+'.mp4'),start)
            assert times==other_times
            x=torch.from_numpy(inp.transpose(0,3,1,2).copy()).float()/255
            began=time.perf_counter()
            with torch.inference_mode():
                output=denoise_seq_fastdvdnet(x,torch.tensor([10/255]),5,model)
            seconds=time.perf_counter()-began
            out=(output.permute(0,2,3,1).numpy().clip(0,1)*255).round().astype(np.uint8)
            name=source.stem+'__'+level
            save_video(dest/(name+'__input.mp4'),inp,rate)
            save_video(dest/(name+'__fastdvdnet.mp4'),out,rate)
            np.savez_compressed(dest/(name+'.npz'),reference=ref,degraded=inp,restored=out,timestamps=times)
            canvas=Image.new('RGB',(768,164),'white');d=ImageDraw.Draw(canvas)
            for j,(title,array) in enumerate(zip(['Reference (resized)','Degraded (resized)','FastDVDnet'],[ref,inp,out])):
                canvas.paste(Image.fromarray(array[len(array)//2]),(j*256,20));d.text((j*256+3,3),title,fill='black')
            canvas.save(dest/(name+'.png'))
            record={'sample_id':source.stem,'level':level,'model':'FastDVDnet','device':'cpu',
                    'source_start_seconds':start,'source_frame_timestamps':times,'frame_count':len(inp),
                    'resolution':[256,144],'noise_map_sigma':10,'added_noise':False,
                    'wall_seconds':seconds,'input_quality':measure(ref,inp),'output_quality':measure(ref,out),
                    'metrics_domain':'uint8 RGB arrays before video encoding, resized reference',
                    'conditioning_audio':None,'saved_video_audio':None,
                    'downstream_audio_rule':'use original source segment at source_start_seconds',
                    'checkpoint_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),
                    'code_commit':subprocess.check_output(['git','-C',str(EXTERNAL),'rev-parse','HEAD'],text=True).strip(),
                    'limitation':'Denoising model applied outside its Gaussian-noise training task; short resized excerpts only.'}
            records.append(record);print(name,seconds,record['input_quality'],record['output_quality'],flush=True)
            (ROOT/'results'/'video_baseline.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()

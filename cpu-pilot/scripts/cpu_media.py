"""Streaming CPU degradations and full-clip audits; no restoration claims."""
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import av
import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
LEVELS = {'moderate': {'sigma': 0.8, 'scale': 2, 'crf': 28},
          'severe': {'sigma': 1.6, 'scale': 4, 'crf': 37}}

def audit(path):
    with av.open(str(path)) as c:
        streams = []
        for s in c.streams:
            row = {'type': s.type, 'codec': s.codec_context.name,
                   'time_base': str(s.time_base), 'start_time': s.start_time,
                   'duration_seconds': float(s.duration*s.time_base) if s.duration is not None else None}
            if s.type == 'video':
                row.update(width=s.width, height=s.height, rate=str(s.average_rate))
            if s.type == 'audio':
                row.update(sample_rate=s.codec_context.sample_rate, channels=s.codec_context.channels)
            streams.append(row)
    with av.open(str(path)) as c:
        timestamps = [float(f.pts*f.time_base) for f in c.decode(video=0)]
    with av.open(str(path)) as c:
        h = hashlib.sha256()
        for p in c.demux(audio=0):
            if p.size:
                h.update(bytes(p))
    return {'streams': streams, 'decoded_frames': len(timestamps),
            'timestamps': timestamps, 'audio_packet_sha256': h.hexdigest()}

def degrade(source, destination, cfg):
    with av.open(str(source)) as inp, av.open(str(destination), 'w') as out:
        v = inp.streams.video[0]
        vo = out.add_stream('libx264', rate=v.average_rate)
        vo.width, vo.height, vo.pix_fmt = v.width, v.height, 'yuv420p'
        vo.time_base = v.time_base
        vo.codec_context.thread_count = 2
        vo.options = {'crf': str(cfg['crf']), 'preset': 'veryfast'}
        audio = {s.index: out.add_stream_from_template(s) for s in inp.streams.audio}
        for packet in inp.demux():
            if packet.stream.type == 'audio' and packet.dts is not None:
                packet.stream = audio[packet.stream.index]
                out.mux(packet)
            elif packet.stream.type == 'video':
                for frame in packet.decode():
                    x = frame.to_ndarray(format='bgr24')
                    x = cv2.GaussianBlur(x, (0, 0), cfg['sigma'])
                    x = cv2.resize(x, (max(2,v.width//cfg['scale']), max(2,v.height//cfg['scale'])), interpolation=cv2.INTER_AREA)
                    x = cv2.resize(x, (v.width,v.height), interpolation=cv2.INTER_LINEAR)
                    f = av.VideoFrame.from_ndarray(x, format='bgr24')
                    f.pts, f.time_base = frame.pts, frame.time_base
                    for p in vo.encode(f): out.mux(p)
        for p in vo.encode(): out.mux(p)

def metrics(source, candidate):
    mse, ssim, motion = [], [], []
    prev_a = prev_b = None
    with av.open(str(source)) as a, av.open(str(candidate)) as b:
        ia, ib = a.decode(video=0), b.decode(video=0)
        while True:
            fa, fb = next(ia, None), next(ib, None)
            if fa is None and fb is None: break
            if fa is None or fb is None: raise ValueError('Frame count mismatch')
            if abs(float(fa.pts*fa.time_base-fb.pts*fb.time_base)) > 0.001:
                raise ValueError('Timestamp drift')
            x,y = fa.to_ndarray(format='rgb24'),fb.to_ndarray(format='rgb24')
            if x.shape != y.shape: raise ValueError('Shape mismatch')
            mse.append(float(np.mean((x.astype(np.float32)-y.astype(np.float32))**2)))
            gx,gy = cv2.cvtColor(x,cv2.COLOR_RGB2GRAY),cv2.cvtColor(y,cv2.COLOR_RGB2GRAY)
            ssim.append(float(structural_similarity(gx,gy,data_range=255)))
            if prev_a is not None:
                motion.append(float(np.mean(np.abs((gx.astype(float)-prev_a)-(gy.astype(float)-prev_b)))))
            prev_a,prev_b=gx.astype(float),gy.astype(float)
    error=float(np.mean(mse))
    return {'frames_evaluated':len(mse),'rgb_psnr_db':10*math.log10(255**2/error) if error else None,
            'rgb_rmse':math.sqrt(error),'gray_ssim_mean':float(np.mean(ssim)),
            'temporal_difference_error_gray_mae':float(np.mean(motion)) if motion else None}

def sheet(paths, output):
    canvas=Image.new('RGB',(960,3*210),'white');draw=ImageDraw.Draw(canvas)
    for row,path in enumerate(paths):
        with av.open(str(path)) as c:
            n=c.streams.video[0].frames
            targets={int((n-1)*q) for q in (0.1,0.35,0.6,0.85)}
            col=0
            for i,f in enumerate(c.decode(video=0)):
                if i in targets:
                    im=f.to_image();im.thumbnail((240,180))
                    canvas.paste(im,(col*240,row*210+25))
                    draw.text((col*240+3,row*210+3),f'{path.stem[-22:]} {float(f.time):.2f}s',fill='black')
                    col+=1
    canvas.save(output)

def main():
    cv2.setNumThreads(2)
    dest=ROOT/'data'/'processed';dest.mkdir(parents=True,exist_ok=True)
    qa=ROOT/'data'/'review';qa.mkdir(parents=True,exist_ok=True)
    rows=[]
    for source in sorted((ROOT/'data'/'raw').glob('*.mp4')):
        original=audit(source);outputs=[source]
        for level,cfg in LEVELS.items():
            target=dest/(source.stem+'__'+level+'.mp4');start=time.perf_counter()
            degrade(source,target,cfg)
            checked=audit(target)
            assert original['decoded_frames']==checked['decoded_frames']
            assert original['audio_packet_sha256']==checked['audio_packet_sha256']
            assert [s for s in original['streams'] if s['type']=='audio'] == [s for s in checked['streams'] if s['type']=='audio']
            assert max(abs(x-y) for x,y in zip(original['timestamps'],checked['timestamps'])) < 0.001
            row={'sample_id':source.stem,'condition':'C1','level':level,'parameters':cfg,
                 'source_audit':original,'output_audit':checked,'metrics':metrics(source,target),
                 'wall_seconds':time.perf_counter()-start,'restoration_executed':False,
                 'conditioning_audio':None,'downstream_audio':'original_source_audio',
                 'ground_truth_verified':False}
            rows.append(row);outputs.append(target)
            print(source.stem,level,row['metrics'],flush=True)
        sheet(outputs,qa/(source.stem+'.jpg'))
    (ROOT/'results'/'cpu_degradations.json').write_text(json.dumps(rows,indent=2))
    env={'os':platform.platform(),'python':platform.python_version(),'pyav':av.__version__,
         'ffmpeg_libraries':av.library_versions,'opencv':cv2.__version__,'numpy':np.__version__}
    (ROOT/'results'/'media_environment.json').write_text(json.dumps(env,indent=2))

if __name__=='__main__':main()

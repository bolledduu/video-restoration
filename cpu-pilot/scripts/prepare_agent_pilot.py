"""Prepare outcome-blind balanced sources for the fixed agent pilot."""
import csv, io, json, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import wavfile
from prepare_unitalk_cpu import fetch
from quick_temporal_pilot import sha, ROOT

def main():
    p=json.loads((ROOT/'agent_pilot_protocol.json').read_text())
    out=ROOT/'data/agent_pilot';out.mkdir(exist_ok=True)
    base=ROOT/'data/unitalk'
    used={r['source'] for r in json.loads((base/'manifest.json').read_text())['records']}
    used|={r['source'] for r in json.loads((ROOT/'data/quick_temporal/manifest.json').read_text())['records']}
    url=f"https://huggingface.co/datasets/{p['dataset']}/resolve/{p['revision']}/"
    records=[];audit=[]
    entries=json.loads((base/'archive_listing.json').read_text(encoding='utf-8-sig'))
    for item in sorted(entries,key=lambda x:(x['size'],x['path'])):
        source=Path(item['path']).stem
        if source in used:continue
        meta=fetch(url+f'csv/val/{source}.csv',base/(source+'.csv'))
        groups=defaultdict(list)
        for r in csv.reader(meta.read_text().splitlines()):groups[r[7]].append(r)
        options={0:[],1:[]}
        for entity,rows in groups.items():
            rows.sort(key=lambda r:float(r[1]))
            for k in range(len(rows)-24):
                w=rows[k:k+25];label=int(w[0][8])
                if label in options and len({r[8] for r in w})==1 and np.allclose(np.diff([float(r[1]) for r in w]),.04,atol=1e-4):
                    options[label].append((float(w[0][1]),entity,w,float(rows[0][1])))
        if not all(options.values()):
            audit.append(dict(source=source,status='no_balanced_windows'));continue
        archive=fetch(url+item['path'],base/(source+'_video.zip'))
        assert archive.stat().st_size==item['size']
        chosen=[]
        with zipfile.ZipFile(archive) as z:
            names={(n.split('/')[-2],round(float(Path(n).stem),4)):n for n in z.namelist() if n.endswith('.jpg')}
            for label in (0,1):
                for start,entity,w,origin in sorted(options[label],key=lambda x:(x[0],x[1])):
                    imgs=[]
                    for r in w:
                        n=names.get((entity,round(float(r[1]),4)))
                        if not n:break
                        im=Image.open(io.BytesIO(z.read(n))).convert('RGB')
                        if min(im.size)<32:break
                        imgs.append(np.asarray(im.resize((128,128),Image.Resampling.BICUBIC)))
                    if len(imgs)==25:
                        chosen.append((start,entity,origin,label,np.stack(imgs)));break
        if len(chosen)!=2:
            audit.append(dict(source=source,status='insufficient_native_resolution'));continue
        az=fetch(url+f'clips_audios/val/{source}.zip',base/(source+'_audio.zip'))
        with zipfile.ZipFile(az) as z:
            for start,entity,origin,label,ref in chosen:
                n=next(n for n in z.namelist() if Path(n).stem==entity and n.endswith('.wav'))
                rate,pcm=wavfile.read(io.BytesIO(z.read(n)))
                assert rate==16000 and pcm.dtype==np.int16 and pcm.ndim==1
                offset=round((start-origin)*rate);audio=pcm[offset:offset+16000].astype(np.float32)/32768
                assert len(audio)==16000
                sid=f'{source}__label{label}';path=out/(sid+'.npz')
                np.savez_compressed(path,reference=ref,audio=audio,labels=np.full(25,label,dtype=np.int64))
                records.append(dict(id=sid,source=source,start=start,entity=entity,label=label,path=str(path),sha256=sha(path),csv_sha256=sha(meta),video_sha256=sha(archive),audio_sha256=sha(az)))
        audit.append(dict(source=source,status='selected'))
        print('Selected',source,flush=True)
        if len(records)==2*p['source_count']:break
    assert len(records)==2*p['source_count']
    result=dict(protocol_sha256=sha(ROOT/'agent_pilot_protocol.json'),records=records,excluded_sources=sorted(used),audit=audit)
    (out/'manifest.json').write_text(json.dumps(result,indent=2))
    print('Preparation complete',len(records),flush=True)

if __name__=='__main__':main()

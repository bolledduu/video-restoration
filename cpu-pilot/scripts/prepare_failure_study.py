"""Freeze a new-source confirmation queue using annotations, never predictions."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import requests

P=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime-root',type=Path,required=True);root=parser.parse_args().runtime_root
    config=json.loads((root/'unitalk_cpu_protocol.json').read_text())
    cached=root/'data/unitalk';listing=cached/'archive_listing.json'
    entries=json.loads(listing.read_text(encoding='utf-8-sig'))
    excluded=sorted(f.stem for f in cached.glob('*.csv'))
    seed='quality-attribution-confirmation-v1'
    archives=[e for e in entries if e['path'].endswith('.zip') and Path(e['path']).stem not in excluded]
    archives.sort(key=lambda e:hashlib.sha256((seed+Path(e['path']).stem).encode()).hexdigest())
    selected=archives[:12];assert len(selected)==12
    folder=P/'data/failure_confirmation';folder.mkdir(exist_ok=True)
    protocol_path=P/'failure_confirmation_protocol.json'
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset':config['dataset'],'revision':config['revision'],'split':'val',
        'selection':'First 12 sources in SHA256(seed+source) order excluding all nine locally cached pilot sources. Fixed source list; no replacement for outcomes or missing eligibility.',
        'seed':seed,'excluded_sources':excluded,'selected_sources':[Path(a['path']).stem for a in selected],
        'listing_sha256':digest(listing),'context_frames':100,'fps':25,
        'track_selection':'Earliest contiguous 100-frame track per category, tie by entity. Speaking: all100 label1; non-speaking: all100 label0; transition:20..80 label1. Same-track windows may not overlap. Select by published annotations; then verify all images/audio/minimum native crop64 before inference. Record exclusions; do not change sources.',
        'primary_matrix':{'noise_sigma':[10,25],'restorers':['fastdvdnet_matched_sigma','spatial_nlm_fixed_h10_template7_search21'],'perception':'TalkNet TalkSet checkpoint, frozen CPU interface, 0.5 threshold','audio':'unaltered, identical across candidates'},
        'primary_outcome':'Per-source rate of captured-correct/restored-wrong timestamps jointly with MUSIQ increase and BRISQUE decrease. Denominator all captured-correct scored timestamps; report conditional and joint rates separately.',
        'secondary':['All introduced errors','Captured-error corrections','Per-source net error change','Both quality deltas individually','Equal-area upper/lower interventions','Temporal residual risk scores vs confidence/disagreement'],
        'statistics':'Report all source results plus source-cluster bootstrap 95% intervals, 10000 resamples, seed20261016. Frames, conditions and agent repetitions are not independent subjects. No threshold tuning on confirmation data.',
        'quality':'MUSIQ and BRISQUE on five equally spaced frames per100-frame clip. SSIM to imperfect source crops diagnostic only. No human preference claim.',
        'agent_phase':'After local measurements finish, run Qwen on every eligible captured/restored condition with frozen two-repeat tool-only prompt; record abstention and API failure rate. Do not select only harmful cases. No claim until responses complete.',
        'scope':'Independent source confirmation of current interface, not new-dataset or unseen-restorer generalization. No learned method or mouth-specific framework is assumed.',
        'capacity_max':{'tracks':36,'degraded_inputs':72,'restored_outputs':144,'qwen_sessions_two_repeats':432},
        'expansion_rule':'After this fixed batch, assess effect size and uncertainty once. Do not keep adding samples until positive. Method work requires additional development sources and a separately untouched test set.',
        'preflight_required':['Verify archives and audio/label alignment','Verify crop eligibility before model calls','Audit checkpoint and reference-input performance','Freeze expanded agent prompt and tool payload hash before agent calls']}
    if protocol_path.exists():
        old=json.loads(protocol_path.read_text());assert old['selected_sources']==protocol['selected_sources'];protocol=old
    else:protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    records=[]
    for archive in selected:
        source=Path(archive['path']).stem;file=folder/(source+'.csv')
        if not file.exists():
            url=f"https://huggingface.co/datasets/{config['dataset']}/resolve/{config['revision']}/csv/val/{source}.csv"
            r=requests.get(url,timeout=60);r.raise_for_status();file.write_bytes(r.content)
        groups=defaultdict(list)
        for row in csv.reader(file.read_text().splitlines()):groups[row[7]].append(row)
        options=defaultdict(list)
        for entity,track in groups.items():
            track.sort(key=lambda r:float(r[1]));times=np.array([float(r[1]) for r in track]);labels=np.array([int(r[8]) for r in track])
            if len(track)<100:continue
            sums=np.convolve(labels,np.ones(100,dtype=int),'valid')
            good_labels=np.convolve((~np.isin(labels,[0,1])).astype(int),np.ones(100,dtype=int),'valid')==0
            contiguous=np.convolve((abs(np.diff(times)-.04)>1e-4).astype(int),np.ones(99,dtype=int),'valid')==0
            for kind in ['speaking','non_speaking','transition']:
                valid=sums==100 if kind=='speaking' else sums==0 if kind=='non_speaking' else (sums>=20)&(sums<=80)
                for k in np.flatnonzero(valid&good_labels&contiguous):
                    options[kind].append(dict(source=source,entity=entity,start=float(times[k]),end=float(times[k+99]),kind=kind,speaking_frames=int(sums[k]),track_origin=float(times[0])))
        chosen=[]
        for kind in ['speaking','non_speaking','transition']:
            eligible=[o for o in sorted(options[kind],key=lambda o:(o['start'],o['entity'])) if not any(o['entity']==c['entity'] and max(o['start'],c['start'])<=min(o['end'],c['end']) for c in chosen)]
            if eligible:chosen.append(eligible[0])
        records.append(dict(source=source,video_archive=archive['path'],video_bytes=archive['size'],annotation_sha256=digest(file),annotation_tracks=chosen,
            missing_categories=sorted(set(['speaking','non_speaking','transition'])-{c['kind'] for c in chosen}),status='ANNOTATION_SELECTED_MEDIA_NOT_VERIFIED'))
        print(source,len(chosen),'annotation windows selected',flush=True)
    manifest=dict(status='ANNOTATION_QUEUE_READY_NOT_EXPERIMENT_RESULTS',protocol_sha256=digest(protocol_path),selection_script_sha256=digest(__file__),records=records,
        selected_tracks=sum(len(r['annotation_tracks']) for r in records),video_download_bytes=sum(r['video_bytes'] for r in records))
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['status','selected_tracks','video_download_bytes']}))

if __name__=='__main__':main()

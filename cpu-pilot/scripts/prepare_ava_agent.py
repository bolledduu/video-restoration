"""Prepare a frozen AVA validation batch; selection never uses model outcomes."""
import argparse, collections, csv, hashlib, io, json, tarfile, urllib.request
from pathlib import Path

LABEL_URL = 'https://research.google.com/ava/download/ava_activespeaker_val_v1.0.tar.bz2'
LIST_URL = 'https://s3.amazonaws.com/ava-dataset/annotations/ava_speech_file_names_v1.txt'
SEED = 'ava-evidence-agent-v1'
LABELS = {'SPEAKING_AUDIBLE': 1, 'NOT_SPEAKING': 0, 'SPEAKING_NOT_AUDIBLE': None}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(1024 * 1024): h.update(b)
    return h.hexdigest()

def download(url, path):
    path = Path(path)
    if path.exists(): return
    tmp = path.with_suffix(path.suffix + '.tmp')
    with urllib.request.urlopen(url, timeout=45) as response, tmp.open('wb') as f:
        while b := response.read(1024 * 1024): f.write(b)
    tmp.replace(path)

def select(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        if r[6] not in LABELS: raise ValueError('Unknown official label')
        groups[r[7]].append(r)
    candidates = collections.defaultdict(list)
    for entity, track in sorted(groups.items()):
        track.sort(key=lambda r: float(r[1]))
        for i, first in enumerate(track):
            start = float(first[1]); window = []
            for row in track[i:]:
                if float(row[1]) >= start + 4: break
                window.append(row)
            if len(window) < 80 or float(window[-1][1]) < start + 3.9: continue
            times = [float(r[1]) for r in window]
            if any(not 0 < b-a <= .1 for a,b in zip(times,times[1:])): continue
            states = [LABELS[r[6]] for r in window]
            if None in states: continue  # preserve label, exclude from binary batch explicitly
            fraction = sum(states)/len(states)
            category = 'speaking' if fraction == 1 else 'listener' if fraction == 0 else 'transition' if .2 <= fraction <= .8 else None
            if category:
                candidates[category].append(dict(entity=entity,start=start,end=start+4,category=category,annotations=window))
    chosen=[]
    for cat in ('speaking','listener','transition'):
        for item in sorted(candidates[cat], key=lambda v:(v['start'],v['entity'])):
            if any(v['entity']==item['entity'] and max(v['start'],item['start']) < min(v['end'],item['end']) for v in chosen): continue
            chosen.append(item);break
    return chosen

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--download-media',action='store_true');args=ap.parse_args()
    root=args.data_root;root.mkdir(parents=True,exist_ok=True)
    archive=root/'ava_activespeaker_val_v1.0.tar.bz2';download(LABEL_URL,archive)
    listing=root/'ava_speech_file_names_v1.txt';download(LIST_URL,listing)
    files={Path(n).stem:n for n in listing.read_text().splitlines() if n.strip()}
    source_rows={}
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith('.csv'):
                rows=list(csv.reader(io.StringIO(tar.extractfile(member).read().decode())))
                source_rows[rows[0][0]]=rows
    order=sorted(source_rows,key=lambda x:hashlib.sha256((SEED+x).encode()).hexdigest())
    selected=order[:4];episodes=[]; exclusions=[]
    for source in selected:
        chosen=select(source_rows[source])
        for c in chosen:
            c['source']=source;c['episode_id']=hashlib.sha256((source+c['entity']+str(c['start'])).encode()).hexdigest()[:16]
            episodes.append(c)
        exclusions += [dict(source=source,category=cat,reason='No eligible nonoverlapping window') for cat in ('speaking','listener','transition') if cat not in {x['category'] for x in chosen}]
    manifest=dict(dataset='AVA-ActiveSpeaker',release='v1.0',split='validation',purpose='First dataset integration batch, not full benchmark or population confirmation',seed=SEED,annotation_sha256=sha(archive),listing_sha256=sha(listing),source_order=order,selected_sources=selected,selection='First four SHA256(seed+source)-ordered sources; earliest eligible 4s speaking/listener/transition windows; no outcome-based replacement',label_mapping=LABELS,inaudible_policy='Preserve original annotations; exclude inaudible speech from this binary batch',episodes=episodes,exclusions=exclusions,status='annotations_ready_media_not_validated')
    path=root/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest: raise RuntimeError('Frozen manifest differs; use a new version instead of overwriting')
    path.write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'sources':selected,'episodes':len(episodes),'exclusions':exclusions}),flush=True)
    if args.download_media:
        audit=[]
        for source in selected:
            dest=root/files[source];url='https://s3.amazonaws.com/ava-dataset/trainval/'+files[source]
            try:
                download(url,dest)
                entry=dict(source=source,path=str(dest.resolve()),url=url,bytes=dest.stat().st_size,sha256=sha(dest),status='downloaded_not_alignment_validated')
            except Exception as exc: entry=dict(source=source,url=url,status='download_failed',error=type(exc).__name__+': '+str(exc)[:300])
            audit.append(entry);(root/'media_download_audit.json').write_text(json.dumps(audit,indent=2));print(entry['source'],entry['status'],flush=True)

if __name__=='__main__':main()

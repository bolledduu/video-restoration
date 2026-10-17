"""Freeze AVA source partitions and annotation-only clip selection before new inference."""
import argparse,csv,hashlib,json
from pathlib import Path
from prepare_ava_agent import select

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    manifest=args.data_root/'download_manifest.json';d=json.loads(manifest.read_text())
    train=d['ava']['dev_sources'];remaining=d['ava']['test_sources']
    ordered=sorted(remaining,key=lambda x:hashlib.sha256(('cts-source-split-v1:'+x).encode()).hexdigest())
    splits={'train':train,'calibration':ordered[:5],'test':ordered[5:]}
    if sum(map(len,splits.values()))!=len(set(sum(splits.values(),[]))):raise ValueError('Source overlap')
    media=json.loads((args.data_root/'ava/merged_v2/manifest.json').read_text())
    paths={r['source']:r['path'] for r in media['sources']};cases=[];exclusions=[]
    for split,sources in splits.items():
        for source in sources:
            files=list((args.data_root/'ava/annotations').rglob(source+'-activespeaker.csv'))
            if len(files)!=1:raise ValueError('Missing or ambiguous annotations: '+source)
            with files[0].open() as f:rows=list(csv.reader(f))
            chosen=select(rows)
            categories={c['category'] for c in chosen}
            for missing in {'speaking','listener','transition'}-categories:exclusions.append(dict(source=source,category=missing,reason='No qualifying annotation-only window'))
            for c in chosen:
                cid=hashlib.sha256(f"{source}:{c['entity']}:{c['start']:.5f}".encode()).hexdigest()[:16]
                cases.append(dict(id=cid,source=source,split=split,media_path=paths[source],annotation_sha256=sha(files[0]),**c))
    result=dict(status='FROZEN_NOT_EXECUTED',scope='New AVA source-disjoint evaluation protocol; full validation only after execution',
        split_seed='cts-source-split-v1',splits=splits,download_manifest_sha256=sha(manifest),code_sha256=sha(Path(__file__)),
        sampling='Earliest eligible four-second track per speaking/listener/transition category; fixed native-resolution/alignment checks may exclude at decode time; never replace based on model outcomes',
        degradation={'synthetic_gaussian_sigma':[15,25,40],'seed':20261017,'quantization':'clip and round uint8; empirical regime'},
        primary_restorer='FastDVDnet',primary_detector='TalkNet',
        limitations=['All previous pilot sources are development-only','Person-to-utterance attribution requires separately verified mappings; not supplied by these ASD labels','No unseen restorer/detector transfer claim from this protocol alone'],
        cases=cases,exclusions=exclusions)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    from collections import Counter
    print('sources', {k:len(v) for k,v in splits.items()},'episodes',dict(Counter(c['split'] for c in cases)))
if __name__=='__main__':main()

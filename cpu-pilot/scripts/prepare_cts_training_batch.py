"""Materialize a frozen CTS split as input to the existing real AVA measurement runner."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--split',choices=['train','calibration','test'],required=True)
    ap.add_argument('--source');ap.add_argument('--exclude-sources',nargs='*',default=[]);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.split!='train':raise ValueError('Calibration/test materialization requires a frozen fitted method; this development entry point permits train only')
    protocol=json.loads(args.protocol.read_text());sources=protocol['splits'][args.split]
    if args.source:
        if args.source not in sources:raise ValueError('Source is not in requested split')
        sources=[args.source]
    sources=[v for v in sources if v not in args.exclude_sources]
    if not sources:raise ValueError('No eligible training sources')
    cases=[c for c in protocol['cases'] if c['source'] in sources]
    episodes=[{**c,'episode_id':c['id']} for c in cases]
    args.output.mkdir(parents=True)
    (args.output/'manifest.json').write_text(json.dumps(dict(selected_sources=sources,episodes=episodes,split=args.split,parent_protocol_sha256=sha(args.protocol)),indent=2))
    receipts=[]
    for source in sources:
        path=next(c['media_path'] for c in cases if c['source']==source)
        receipts.append(dict(source=source,path=path,sha256=sha(path)))
    (args.output/'media_download_audit.json').write_text(json.dumps(receipts,indent=2))
    print('Prepared training-only sources',sources,'episodes',len(episodes))
if __name__=='__main__':main()

"""Record local dependencies without credentials, for the Antigravity handoff."""
import argparse, hashlib, importlib.metadata, json, platform, shutil, subprocess
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1048576):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    root=args.runtime_root.resolve();out=args.output;out.mkdir(parents=True,exist_ok=True)
    selected=[]
    for relative in ['scripts/evaluate_causal.py','scripts/prepare_causal.py','scripts/run_causal_omnivr.py']:
        path=root/relative;selected.append({'path':relative,'sha256':digest(path)})
    for folder in ('external/fastdvdnet','external/TalkNet-ASD/model'):
        for path in sorted((root/folder).rglob('*.py')):selected.append({'path':path.relative_to(root).as_posix(),'sha256':digest(path)})
    packages={}
    paths=[str(root/p) for p in ('tools/quality_cpu','tools/iqa_models','tools/python','tools/media_runtime')]
    for dist in list(importlib.metadata.distributions())+list(importlib.metadata.distributions(path=paths)):
        name=dist.metadata.get('Name')
        if name:packages.setdefault(name,set()).add(dist.version)
    revision=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    receipt={'runtime_root':str(root),'git_revision':revision,'python':platform.python_version(),'platform':platform.platform(),
        'source_files':selected,'packages':{k:sorted(v) for k,v in sorted(packages.items())},
        'note':'Inventory of existing local runtime; distribution versions may coexist in vendored paths. Media and weights remain local.'}
    (out/'runtime_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    for name in ('manifest.json','media_download_audit.json'):
        shutil.copyfile(root/'data/ava_agent'/name,out/('dataset_'+name))
    print('Runtime and dataset receipts saved')

if __name__=='__main__':main()

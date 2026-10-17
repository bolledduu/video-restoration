"""Merge separately downloaded AVA streams, preserving the source components."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--ffmpeg',required=True)
    ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    data=json.loads(args.audit.read_text());root=Path(data['root'])
    groups={}
    for r in data['files']:
        if Path(r['path']).parts[:2] != ('ava','videos'):continue
        source=Path(r['path']).name.split('.')[0];groups.setdefault(source,[]).append(r)
    args.output_dir.mkdir(parents=True,exist_ok=True);records=[]
    for source,rows in sorted(groups.items()):
        complete=[r for r in rows if r['video'] and r['audio']]
        if complete:
            records.append(dict(source=source,status='existing_av',path=str(root/complete[0]['path'])));continue
        video=[r for r in rows if r['video']];audio=[r for r in rows if r['audio']]
        if len(video)!=1 or len(audio)!=1:raise ValueError('Ambiguous stream components: '+source)
        out=args.output_dir/(source+'.mkv')
        if out.exists():raise FileExistsError(out)
        inputs=[root/video[0]['path'],root/audio[0]['path']]
        command=[args.ffmpeg,'-hide_banner','-loglevel','error','-n','-i',str(inputs[0]),'-i',str(inputs[1]),
            '-map','0:v:0','-map','1:a:0','-c','copy',str(out)]
        subprocess.run(command,check=True,capture_output=True)
        records.append(dict(source=source,status='remuxed_original_streams',path=str(out),inputs=[str(p) for p in inputs],
            sha256=hashlib.file_digest(out.open('rb'),'sha256').hexdigest()))
    manifest=args.output_dir/'manifest.json'
    if manifest.exists():raise FileExistsError(manifest)
    manifest.write_text(json.dumps(dict(scope='AV stream availability; annotation alignment still requires validation',sources=records),indent=2))
    print('Sources:',len(records),'remuxed:',sum(r['status'].startswith('remuxed') for r in records))
if __name__=='__main__':main()

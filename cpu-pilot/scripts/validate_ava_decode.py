"""Check one annotation-timed AV decode per source; no model outcome selection."""
import argparse,csv,json,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--annotations',type=Path,required=True);ap.add_argument('--ffmpeg',required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    records=json.loads(args.manifest.read_text())['sources']
    def verify(record):
        sid=record['source'];files=list(args.annotations.rglob(sid+'-activespeaker.csv'))
        if len(files)!=1:return dict(source=sid,status='ANNOTATION_MISSING_OR_AMBIGUOUS')
        with files[0].open() as f:annotations=list(csv.reader(f))
        labels={r[6] for r in annotations};start=min(float(r[1]) for r in annotations)
        cmd=[args.ffmpeg,'-hide_banner','-nostats','-v','info','-ss',str(start),'-i',record['path'],
             '-t','1','-map','0:v:0','-map','0:a:0','-f','null','-']
        result=subprocess.run(cmd,capture_output=True,text=True,errors='replace',timeout=90)
        import re
        frames=re.findall(r'frame=\s*(\d+)',result.stderr)
        count=int(frames[-1]) if frames else 0
        return dict(source=sid,start=start,annotation_rows=len(annotations),labels=sorted(labels),
                    status='DECODE_OK' if result.returncode==0 and count>0 else 'DECODE_FAILED',decoded_frames=count)
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(verify,records))
    report=dict(scope='One second at first annotated time; technical decode only, not semantic synchronization validation',sources=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    from collections import Counter
    print(Counter(r['status'] for r in rows))
if __name__=='__main__':main()

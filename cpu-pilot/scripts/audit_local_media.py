"""Audit local media stream presence without modifying downloaded files."""
import argparse, json, re, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--ffmpeg',required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    files=sorted(p for p in args.root.rglob('*') if p.suffix.lower() in ('.mp4','.mkv','.webm','.avi','.wav'))
    def inspect(p):
        result=subprocess.run([args.ffmpeg,'-hide_banner','-i',str(p)],capture_output=True,text=True,errors='replace',timeout=30)
        log=result.stderr
        streams=[line.strip() for line in log.splitlines() if re.search(r'Stream #\d+:\d+',line)]
        return dict(path=str(p.relative_to(args.root)),bytes=p.stat().st_size,
                    video=any('Video:' in s for s in streams),audio=any('Audio:' in s for s in streams),
                    streams=streams,probe_readable=bool(streams))
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(inspect,files))
    summary=dict(files=len(rows),unreadable=sum(not r['probe_readable'] for r in rows),
        video_and_audio=sum(r['video'] and r['audio'] for r in rows),
        video_only=sum(r['video'] and not r['audio'] for r in rows),
        audio_only=sum(r['audio'] and not r['video'] for r in rows))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(root=str(args.root),summary=summary,
        scope='Stream headers only; does not certify decoding, synchronization, annotations or ground truth',files=rows),indent=2))
    print(summary)
if __name__=='__main__':main()

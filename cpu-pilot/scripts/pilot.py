#!/usr/bin/env python3
"""CPU preprocessing and cautious scoring. No audio-conditioned restorer is implemented."""
import argparse,csv,json,math,subprocess,shutil,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/"manifest.json").read_text())

def run(cmd):
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode: raise RuntimeError(p.stderr[-1000:])
    return p.stdout

def probe(path):
    x=json.loads(run(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(path)]))
    video=next((s for s in x["streams"] if s.get("codec_type")=="video"),None)
    if not video: raise ValueError("No video stream")
    return {"width":int(video["width"]),"height":int(video["height"]),"duration":float(x["format"].get("duration",0)),"has_audio":any(s.get("codec_type")=="audio" for s in x["streams"])}

def fetch():
    from huggingface_hub import hf_hub_download
    for sample in MANIFEST["samples"]:
        dst=ROOT/"data"/"raw"/(sample["sample_id"]+".mp4");dst.parent.mkdir(parents=True,exist_ok=True)
        path=hf_hub_download(repo_id=MANIFEST["dataset"],repo_type="dataset",filename=sample["file"])
        shutil.copyfile(path,dst)
        if not probe(dst)["has_audio"]: raise ValueError("Clip lacks audio: "+str(dst))
        print("downloaded",dst)

def process(source,out,mode):
    p=probe(source);w,h=p["width"],p["height"]
    if mode=="degraded":
        small_w=max(2,((w//4)//2)*2);small_h=max(2,((h//4)//2)*2)
        filter=f"scale={small_w}:{small_h}:flags=area,scale={w}:{h}:flags=bilinear,format=yuv420p";crf="37"
    elif mode=="sharpen_only":filter="unsharp=5:5:0.6,format=yuv420p";crf="18"
    else: raise ValueError(mode)
    out.parent.mkdir(parents=True,exist_ok=True)
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(source),"-vf",filter,"-c:v","libx264","-preset","veryfast","-crf",crf,"-c:a","copy",str(out)])
    return probe(out)

def image_metrics(original,candidate,max_frames=None):
    a=cv2.VideoCapture(str(original));b=cv2.VideoCapture(str(candidate));vals=[]
    try:
        while max_frames is None or len(vals)<max_frames:
            ok1,f1=a.read();ok2,f2=b.read()
            if ok1 != ok2:raise ValueError("Mismatched frame count")
            if not ok1:break
            if f1.shape!=f2.shape:raise ValueError("Mismatched frame shape")
            vals.append(float(np.mean((f1.astype(float)-f2.astype(float))**2)))
    finally:
        a.release();b.release()
    if not vals:raise ValueError("No comparable frames")
    mse=float(np.mean(vals))
    return {"frames_evaluated":len(vals),"psnr_db":round(10*math.log10(255*255/mse),4) if mse>0 else None,"pixel_rmse":round(math.sqrt(mse),4)}

def evaluate(directory=None,output_path=None):
    source_root=Path(directory) if directory else ROOT/"data"/"raw"
    clips=[]
    for s in MANIFEST["samples"]:
        original=source_root/(s["sample_id"]+".mp4")
        if not original.exists():
            clips.append({"sample_id":s["sample_id"],"status":"NOT_DOWNLOADED","research_result":None});continue
        if not probe(original)["has_audio"]:
            clips.append({"sample_id":s["sample_id"],"status":"INVALID_NO_AUDIO"});continue
        degraded=ROOT/"data"/"processed"/(s["sample_id"]+"__degraded.mp4")
        sharp=ROOT/"data"/"processed"/(s["sample_id"]+"__sharpen_only.mp4")
        process(original,degraded,"degraded");process(degraded,sharp,"sharpen_only")
        clips.append({"sample_id":s["sample_id"],"status":"PREPROCESSING_ONLY","degraded":image_metrics(original,degraded),"sharpen_only":image_metrics(original,sharp),"research_result":None})
    result={"research_status":"UNTESTED","input_mode":"ENGINEERING_FIXTURES" if directory else "DATASET_SOURCE","annotation_valid":False,"audio_conditioned_restorer_executed":False,"speaker_detector_executed":False,"clips":clips}
    dest=Path(output_path) if output_path else ROOT/"results"/"processing_results.json";dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result

def score(path):
    valid={"original","degraded","video_only_restored","audio_conditioned_restored"}
    truth={s["sample_id"]:s["published_answer"] for s in MANIFEST["samples"]};pred={}
    with open(path,newline="") as file:
        for r in csv.DictReader(file):
            key=(r["sample_id"],r["condition"])
            if r["sample_id"] not in truth or r["condition"] not in valid or r["prediction"] not in "ABCD" or len(r["prediction"])!=1 or key in pred:raise ValueError("Invalid prediction record")
            pred[key]=r["prediction"]
    required={(s,c) for s in truth for c in valid}
    if set(pred)!=required: raise ValueError("Missing/extra predictions")
    results={c:sum(pred[(s,c)]!=truth[s] for s in truth)/len(truth) for c in valid}
    report={"research_status":"PUBLISHED_MULTIPLE_CHOICE_LABELS_ONLY","ground_truth_independently_verified":False,"causal_mechanism_verified":False,"multiple_choice_error_rates":results,"multiple_choice_error_delta_audio_minus_video_only":results["audio_conditioned_restored"]-results["video_only_restored"],"false_attribution_rate":None,"warning":"Question-answer errors are not frame-level false speaker attributions. Audio provenance and independent labels are not validated by this legacy scorer."}
    out=ROOT/"results"/"speaker_scoring.json";out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("action",choices=["fetch","prepare","score"]);a.add_argument("--input-dir");a.add_argument("--predictions");args=a.parse_args()
    if args.action=="fetch":fetch()
    elif args.action=="prepare":evaluate(args.input_dir)
    else:
        if not args.predictions:a.error("Provide --predictions CSV with actual model outputs")
        score(args.predictions)

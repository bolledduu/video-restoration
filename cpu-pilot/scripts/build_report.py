"""Summarize measured outputs and create a local-only media review page."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import av
import numpy as np
from cpu_media import ROOT

def main():
    sources=json.loads((ROOT/'results/source_downloads.json').read_text())
    degradation=json.loads((ROOT/'results/cpu_degradations.json').read_text())
    baseline=json.loads((ROOT/'results/video_baseline.json').read_text())
    predictions=json.loads((ROOT/'results/talknet_diagnostics.json').read_text())
    assert len(sources)==3 and len(degradation)==6 and len(baseline)==6 and len(predictions)==18
    annotations=[]
    observations=[
      'Four people visible in sampled couch/unboxing frames; camera cuts/zoom, sunglasses and hand occlusion. Published question requires a specific box-removal event; inference excerpt is not verified to cover that event.',
      'Alternating shots of a suited man and a man in a dark short-sleeve shirt; cut within the diagnostic excerpt. Automatic largest-face identity changes are possible.',
      'Office ensemble scene with cuts and occlusion. Late excerpt centers on a light-shirted man, not necessarily the woman named in the question. Do not score this crop using published answer C.']
    for i,s in enumerate(sources):
        p=ROOT/'data/raw'/(s['sample_id']+'.mp4')
        with av.open(str(p)) as c:
            rms=[];peaks=[];samples=0;first=None;last=None
            for f in c.decode(audio=0):
                x=f.to_ndarray().astype(float)
                rms.append(float(np.sqrt(np.mean(x*x))));peaks.append(float(np.max(np.abs(x))))
                if first is None:first=float(f.time)
                last=float(f.time)+f.samples/f.sample_rate;samples+=f.samples
        annotations.append({'sample_id':s['sample_id'],'video_id':s['published_metadata']['video_id'],
          'published_answer':s['published_metadata']['answer'],'source_byte_hash_verified':hashlib.sha256(p.read_bytes()).hexdigest()==s['sha256'],
          'technical_verification':'download path, published record, decoded streams and timing checked',
          'visual_review':'sampled keyframes inspected, not full motion annotation','observations':observations[i],
          'audio_review':'decoded signal inspected numerically; no independent listening/transcript verification',
          'audio_signal':{'first_time_seconds':first,'decoded_end_seconds':last,'samples_per_channel':samples,
                          'mean_frame_rms':float(np.mean(rms)),'peak_absolute_sample':max(peaks)},
          'speaker_states':[],'speech_timestamps':[],'published_answer_independently_verified':False,
          'suitability':'PROVISIONAL; not accepted for causal or accuracy evaluation',
          'natural_offscreen_condition_verified':False,
          'review_decision':'User requested keeping annotations provisional on 2026-10-12',
          'license':'Dataset README body says CC BY-NC 4.0; header says MIT. Source media rights not independently established; no media committed.'})
    (ROOT/'results/provisional_annotations.json').write_text(json.dumps(annotations,indent=2))
    rows=[]
    for r in baseline:
        rows.append({'sample_id':r['sample_id'],'level':r['level'],'frames':r['frame_count'],
                     'input_psnr_db':r['input_quality']['psnr_db'],'output_psnr_db':r['output_quality']['psnr_db'],
                     'input_ssim':r['input_quality']['ssim_rgb_mean'],'output_ssim':r['output_quality']['ssim_rgb_mean'],
                     'cpu_inference_seconds':r['wall_seconds']})
    with (ROOT/'results/quality_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    checks=[]
    for sample in {r['sample_id'] for r in predictions}:
        for level in ['moderate','severe']:
            group=[r for r in predictions if r['sample_id']==sample and r['level']==level]
            assert len(group)==3
            fixed=len({r['audio_sha256'] for r in group})==1
            audio_equal=all(np.array_equal(group[0]['logits_audio_only'],r['logits_audio_only']) for r in group[1:])
            assert fixed and audio_equal
            checks.append({'sample_id':sample,'level':level,'fixed_audio_hash_identical':fixed,'audio_only_logits_identical':audio_equal})
    status={'decision':'NO-GO','decision_scope':'full-paper investment on current evidence',
            'pilot_preparation_decision':'NO','feasibility_assessment':'CLOSED','compute_policy':'laptop only','paid_gpu_request_ready':False,'hypothesis_status':'UNTESTED','audio_conditioned_restorer_executed':False,'video_only_restorer_executed':True,
            'speaker_detector_executed':True,'independently_verified_speaker_labels':False,
            'delta_false_attribution':None,'speaker_accuracy':None,'causal_controls_complete':False,
            'rq1':'testable but untested','rq2':'operational recoverability only; not established',
            'rq3':'defer new method; no measured advantage over simple rules',
            'fixed_audio_checks':checks,'conditions':{'C0':'available','C1':'six full clips','C2':'six short resized FastDVDnet diagnostics; task-mismatched',
            'C3':'BLOCKED: missing genuine audio-conditioned restorer','C4':'BLOCKED: same dependency','C5':'UNVERIFIED'},
            'missing_metrics':['LPIPS','calibration','verified accuracy','false-attribution rate','conditioning-audio sensitivity','abstention performance'],
            'software_tests':'8 passed after fixes; initial 5 passed outside sandbox'}
    supplement=ROOT/'results/omnivr_cpu_evaluation.json'
    if supplement.exists():
        supplemental_records=json.loads(supplement.read_text())
        status['supplemental_hf_evaluation']={
            'model':'xin1u/OmniVR','public_weights_located':True,
            'author_output_clips_evaluated':len(supplemental_records),
            'local_omnivr_restoration_executed':False,
            'conditioning_audio_swaps_executed':False,
            'report':'HF_CONTINUATION.md',
            'interpretation':'CPU evaluation of author-generated outputs; not a completed causal test'}
    (ROOT/'results/research_status.json').write_text(json.dumps(status,indent=2))
    packages={p:importlib.metadata.version(p) for p in ['numpy','opencv-python','av','torch','scipy','scikit-image','Pillow']}
    env={'packages':packages,'cpu':'AMD Ryzen 5 5600H with Radeon Graphics','physical_cores':6,'logical_processors':12,
         'initial_total_visible_ram_kib':7764856,'initial_free_ram_kib':1225820,'cuda_available':False,
         'ffmpeg_cli_on_path':False,'media_backend':'PyAV 16.0.1 bundled FFmpeg libraries; versions in media_environment.json',
         'torch_threads':2,'seed':0}
    (ROOT/'results/laptop_environment.json').write_text(json.dumps(env,indent=2))
    html=['<!doctype html><meta charset="utf-8"><title>CPU pilot review</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;background:#101722;color:#eef3f8}video{width:31%;margin:1%}img{max-width:100%}section{padding:20px;background:#1c2736;margin:20px 0}a{color:#8cceff}</style><h1>CPU pilot: local media review</h1><p>Full-paper investment: NO-GO. Laptop feasibility assessment closed. No further GPU preparation scheduled. Hypothesis untested: no audio-conditioned restoration. Speaking labels remain provisional. Original and degraded clips have authentic audio. Short FastDVDnet previews are silent. Images and videos remain local.</p>']
    for a in annotations:
        sid=a['sample_id'];html.append('<section><h2>'+a['video_id']+'</h2><p>'+a['observations']+'</p><p>Original / moderate degradation / severe degradation</p>')
        for file in ['../raw/'+sid+'.mp4','../processed/'+sid+'__moderate.mp4','../processed/'+sid+'__severe.mp4']:
            html.append('<video controls preload="metadata" src="'+file+'"></video>')
        for level in ['moderate','severe']:
            html.append('<h3>'+level+' — short resized baseline</h3><img src="../baseline/'+sid+'__'+level+'.png"><video controls src="../baseline/'+sid+'__'+level+'__fastdvdnet.mp4"></video>')
        html.append('</section>')
    (ROOT/'data/review/index.html').write_text('\n'.join(html),encoding='utf-8')
    print(json.dumps({'source_clips':len(sources),'degraded_clips':len(degradation),'baseline_runs':len(baseline),'speaker_diagnostics':len(predictions),'decision':status['decision'],'decision_scope':status['decision_scope']},indent=2))

if __name__=='__main__':main()

"""Report all spatial interventions without turning an exploratory screen into confirmation."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parents[1]

def auc(labels,scores):
    a=np.asarray(scores)[np.asarray(labels,dtype=bool)];b=np.asarray(scores)[~np.asarray(labels,dtype=bool)]
    if not len(a) or not len(b):return None
    return float(np.mean((a[:,None]>b).astype(float)+.5*(a[:,None]==b)))

def main():
    path=P/'results/locality_measurements.json';d=json.loads(path.read_text())
    assert d['status']=='COMPLETE' and len(d['cases'])==17
    assert hashlib.sha256((P/'locality_protocol.json').read_bytes()).hexdigest()==d['protocol_sha256']
    assert hashlib.sha256((P/'scripts/localize_restoration_effects.py').read_bytes()).hexdigest()==d['script_sha256']
    by_source=defaultdict(lambda:dict(pairs=0,n=0,captured_errors=0,restored_errors=0,harm=0,corrections=0,lower_rescued=0,upper_rescued=0,lower_errors=0,upper_errors=0))
    rows=[];scores=defaultdict(list);targets=[]
    for c in d['cases']:
        s=c['stats'];lo=c['hybrids']['lower'];up=c['hybrids']['upper'];dest=by_source[c['source']];dest['pairs']+=1
        for k in ['n','captured_errors','restored_errors','harm','corrections']:dest[k]+=s[k]
        for k,v in [('lower_rescued',lo['rescued_restoration_harm']),('upper_rescued',up['rescued_restoration_harm']),('lower_errors',lo['errors']),('upper_errors',up['errors'])]:dest[k]+=v
        rows.append(dict(id=c['id'],source=c['source'],**s,lower_rescued=lo['rescued_restoration_harm'],upper_rescued=up['rescued_restoration_harm'],lower_errors=lo['errors'],upper_errors=up['errors'],**c['features']))
        targets.append(s['harm']>0)
        for k,v in c['features'].items():scores[k].append(v)
        a,b=c['quality']['captured'],c['quality'][c['candidate']]
        scores['musiq_improvement'].append(b['musiq']-a['musiq'])
        scores['brisque_improvement'].append(a['brisque']-b['brisque'])
        cp=np.array(c['predictions']['captured']['AV']);rp=np.array(c['predictions']['restored']['AV'])
        scores['binary_disagreement'].append(float(np.mean((cp>=.5)!=(rp>=.5))))
        scores['restored_uncertainty'].append(float(np.mean(1-2*abs(rp-.5))))
    totals={k:sum(v[k] for v in by_source.values()) for k in next(iter(by_source.values()))}
    supported=[k for k,v in by_source.items() if v['harm']>0 and v['lower_rescued']>v['upper_rescued']]
    passed=len(supported)>=2
    ranking={k:auc(targets,v) for k,v in scores.items()}
    report=dict(status='COMPLETE',measurement_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),totals=totals,by_source=dict(by_source),
        lower_face_gate_passed=passed,supporting_sources=supported,descriptive_candidate_harm_auc=ranking,
        auc_warning='Same 17 exploratory candidates; dependent conditions and tracks, no fitted estimator, no held-out validation or incremental predictive claim.',
        decision='Proceed only to independent lower-face confirmation.' if passed else 'Do not build a lower-face-specific framework on this screen; broader quality-attribution failure study remains separately motivated.')
    (P/'results/locality_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (P/'results/locality_pairs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Local restoration intervention: completed exploratory screen','',
        '**Decision: '+report['decision']+'**','',
        'All 17 cached restoration comparisons were retained: seven tracks from five source videos. This reuses known pilot material and is exploratory. It does not add independent source replication or new Qwen responses.','',
        'We reproduced captured/restored TalkNet scores, then replaced a fixed lower or upper half of the restored crop with captured pixels. Masks have equal area and identical feathering. Audio is identical. Regions are coarse face-crop halves, not verified anatomical mouth masks.','',
        '| Source | Comparisons | Captured errors | Restored errors | New errors | Corrections | New errors rescued: lower | New errors rescued: upper |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for source,v in sorted(by_source.items()):lines.append(f"| {source} | {v['pairs']} | {v['captured_errors']}/{v['n']} | {v['restored_errors']}/{v['n']} | {v['harm']} | {v['corrections']} | {v['lower_rescued']} | {v['upper_rescued']} |")
    lines += ['',f"Totals: {totals['harm']} introduced-error occurrences, {totals['corrections']} corrections. Lower replacement rescues {totals['lower_rescued']} of these errors; upper replacement rescues {totals['upper_rescued']}. These rescues can overlap. All-case hybrid errors are {totals['lower_errors']} (lower retained) and {totals['upper_errors']} (upper retained), compared with {totals['captured_errors']} captured and {totals['restored_errors']} restored.",'',
        f"Frozen exploratory gate: lower replacement rescues more harmful timestamps than upper replacement on at least two sources. Result: {'PASS' if passed else 'FAIL'}; supporting sources: {supported}.",'',
        'Equal-area patches do not guarantee equal perturbation energy, and compositing can introduce artifacts. Spatial intervention does not isolate temporal causality. Source position, pose, occlusion, model context and original detector errors remain alternative explanations. This is a sensitivity experiment, not proof of invented mouth motion.','',
        '## Fixed risk scores (descriptive only)','',
        'Target: any captured-correct/restored-wrong timestamp in a candidate. Higher risk scores were defined before intervention inference. AUROC here uses all 17 cached comparisons and is not a generalization estimate. No direction is flipped after seeing results.','',
        '| Score | Candidate-harm AUROC |','|---|---:|']
    for k,v in ranking.items():lines.append(f"| {k} | {v:.3f} |" if v is not None else f'| {k} | Undefined: missing class |')
    lines += ['', 'No learned assessor, held-out incremental advantage, useful agent safeguard, or paper novelty has been established. Frame counts are repeated opportunities, not independent samples. Full experiments must split by source and report source-level uncertainty.','',
        '## Existing work to distinguish','',
        'LASER already incorporates lip landmarks for speaker-detection robustness. Our proposed question concerns restoration-induced errors and the adequacy of quality assessment; merely attending to lips is not a novelty claim. https://openaccess.thecvf.com/content/WACV2026/html/Nguyen_LASER_Lip_Landmark_Assisted_Speaker_Detection_for_Robustness_WACV_2026_paper.html','',
        'TalkNet is the perception tool used here; Qwen is a separate agent receiving tool measurements. https://github.com/TaoRuijie/TalkNet-ASD','']
    (P/'LOCALITY_SCREEN_DECISION_2026-10-16.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4));names=sorted(by_source);x=np.arange(len(names));width=.25
    for off,key,label in [(-1,'harm','Introduced errors'),(0,'lower_rescued','Rescued: lower retained'),(1,'upper_rescued','Rescued: upper retained')]:
        bars=ax.bar(x+off*width,[by_source[n][key] for n in names],width,label=label);ax.bar_label(bars,padding=2,fontsize=8)
    ax.set_xticks(x,names,rotation=15);ax.set_ylabel('Timestamp occurrences (conditions repeat frames)');ax.set_title('Spatial intervention on TalkNet — not new Qwen results');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(P/'results/locality_intervention.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

"""Audit solution prototypes and timing-response diagnostic, retaining negative findings."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from localize_restoration_effects import sha,stats
from report_locality_screen import auc

P=Path(__file__).resolve().parents[1]

def main():
    file=P/'results/temporal_evidence_measurements.json';d=json.loads(file.read_text());assert d['status']=='COMPLETE' and len(d['cases'])==17
    assert sha(P/'temporal_evidence_protocol.json')==d['protocol_sha256'] and sha(P/'scripts/test_temporal_evidence.py')==d['script_sha256']
    names=['captured','restored','static_residual','dynamic_residual','half_blend'];totals={};by_source={}
    for name in names:
        rows=[];sources=defaultdict(lambda:dict(errors=0,captured_errors=0,n=0))
        for c in d['cases']:
            m=c['metrics'][name];baseline=c['metrics']['captured'];y=np.array(c['labels']);s=stats(baseline['probability'],m['probability'],y)
            assert s['restored_errors']==m['errors']
            rows.append({**s,'musiq_delta':m['musiq']-baseline['musiq'],'brisque_delta':m['brisque']-baseline['brisque'],'ssim_delta':m['ssim']-baseline['ssim'],
                'joint_quality_gain':m['musiq']>baseline['musiq'] and m['brisque']<baseline['brisque'],'clipped_fraction':m['clipped_fraction']})
            src=sources[c['source']];src['errors']+=m['errors'];src['captured_errors']+=baseline['errors'];src['n']+=len(y)
        t={k:sum(r[k] for r in rows) for k in ['n','captured_errors','restored_errors','harm','corrections','joint_quality_gain']}
        t.update({k:float(np.mean([r[k] for r in rows])) for k in ['musiq_delta','brisque_delta','ssim_delta','clipped_fraction']})
        t['sources_worse_than_captured']=[s for s,v in sources.items() if v['errors']>v['captured_errors']]
        t['solution_screen_passed']=t['restored_errors']<t['captured_errors'] and not t['sources_worse_than_captured'] and t['musiq_delta']>0 and t['brisque_delta']<0
        totals[name]=t;by_source[name]=dict(sources)
    targets=[c['original_stats']['harm']>0 for c in d['cases']]
    risk=[c['alignment_response_change'] for c in d['cases']]
    simple=[float(np.mean((np.array(c['metrics']['captured']['probability'])>=.5)!=(np.array(c['metrics']['restored']['probability'])>=.5))) for c in d['cases']]
    scorechange=[float(np.mean(abs(np.array(c['metrics']['captured']['probability'])-np.array(c['metrics']['restored']['probability'])))) for c in d['cases']]
    rankings={'alignment_response_change':auc(targets,risk),'binary_disagreement':auc(targets,simple),'absolute_score_change':auc(targets,scorechange)}
    passed=[k for k,v in totals.items() if v['solution_screen_passed']]
    report={'status':'COMPLETE','measurement_sha256':sha(file),'pairs':17,'sources':5,'detector_calls':153,'new_qwen_calls':0,
        'totals':totals,'by_source':by_source,'solution_screen_passed':passed,'descriptive_harm_auroc':rankings,
        'warning':'Exploratory reused inputs, three fixed prototypes, no held-out advantage. Audio shifts are synthetic mismatch probes and do not certify true synchronization. Quality metrics are not human judgments.'}
    (P/'results/temporal_evidence_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Solution mechanism experiment','',
        f"Completed all 17 cached comparisons from five sources: 153 detector calls and zero new Qwen calls. Prototype gate passes: {passed}.",'',
        '| Video supplied to detector | Errors /633 | New errors vs captured | Corrections vs captured | Mean MUSIQ change | Mean BRISQUE change | Both quality scores improve: pairs /17 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k,t in totals.items():lines.append(f"| {k} | {t['restored_errors']} | {t['harm']} | {t['corrections']} | {t['musiq_delta']:+.3f} | {t['brisque_delta']:+.3f} | {t['joint_quality_gain']} |")
    lines += ['', 'MUSIQ higher and BRISQUE lower are better. Mean deltas average pairs, not independent sources. All variants answer every timestamp; no improvement can arise from abstention. The 633 opportunities repeat source timestamps across restoration conditions.','',
        'Static residual: captured video plus the time-average restoration residual. Before clipping and rounding this preserves captured pixel differences across frames. Dynamic residual: captured video plus only the time-varying residual; this preserves restored pixel differences before clipping. Neither is a semantic appearance/motion decomposition; head motion, noise and clipping are confounds. Half blend is an ordinary control.','',
        'Frozen solution screen: fewer total errors than captured-only, no source worse than captured-only, positive mean MUSIQ change and negative mean BRISQUE change. This is an exploratory engineering gate, not significance, novelty certification, or validation on untouched sources.','',
        '## Audio-timing response test','',
        'For both captured and restored video, rerun the detector with original audio and circular shifts of +/-0.32 seconds. Measure how restoration changes the original-minus-shifted response contrast. Audio-only scores match exactly between captured and restored video for each audio condition. A change in contrast is model sensitivity, not proof of genuine synchronization or invented speech.','',
        '| Risk score | AUROC for any introduced error |','|---|---:|']
    for k,v in rankings.items():lines.append(f'| {k} | {v:.3f} |' if v is not None else f'| {k} | Undefined |')
    lines += ['', 'These AUROCs use the same 17 exploratory pairs. With no beneficial original restorations, disagreement can perfectly rank harmful candidates without demonstrating a useful general solution. The timing test must add value on independent sources containing both improvements and regressions before it can justify a new assessor.','',
        '## Per-source errors','', '| Source | Captured | Restored | Static residual | Dynamic residual | Half blend |','|---|---:|---:|---:|---:|---:|']
    for s in sorted(by_source['captured']):lines.append('| '+s+' | '+' | '.join(str(by_source[k][s]['errors']) for k in names)+' |')
    lines += ['', 'The previously prepared 12-source confirmation queue was not used to design or select these prototypes. No novel framework is claimed from this screen.','']
    (P/'TEMPORAL_SOLUTION_SCREEN_2026-10-16.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

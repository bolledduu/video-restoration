"""Separate a complete detector bound from incomplete agent evidence; never impute missing runs."""
import json
from pathlib import Path
import numpy as np
from report_benefit_screen import pair_stats,sha
P=Path(__file__).resolve().parents[1]

def main():
    mp=P/'results/benefit_measurements.json';m=json.loads(mp.read_text());assert m['status']=='COMPLETE' and len(m['cases'])==9
    detector=[]
    for c in m['cases']:
        assert sha(c['path'])==c['sha256'] and sha(c['outputs'])==c['output_sha256']
        with np.load(c['path']) as b:y=b['labels'][c['indices']]
        a=(np.array(c['candidates']['captured']['probability'])>=.5).astype(int)
        r=(np.array(c['candidates']['restored']['probability'])>=.5).astype(int)
        detector.append({'case_id':c['id'],**pair_stats(a,r,y)})
    qp=P/'results/benefit_qwen.json';q=json.loads(qp.read_text());assert sha(P/'scripts/benefit_qwen.py')==q['script_sha256']
    pairs=[];missing=[]
    for c in q['cases']:
        for repeat in range(2):
            found={name:[r for r in q['runs'] if (r['case_id'],r['repeat'],r['condition'])==(c['id'],repeat,name) and r['status']=='completed'] for name in ['captured','restored']}
            for name,rs in found.items():
                assert len(rs)<=1
                if not rs:missing.append({'case_id':c['id'],'repeat':repeat,'condition':name})
            if not all(found.values()):continue
            with np.load(c['path']) as b:y=b['labels'][c['indices']]
            a,r=[found[name][0]['submission']['frame_tags'] for name in ['captured','restored']]
            pairs.append({'case_id':c['id'],'repeat':repeat,**pair_stats(a,r,y)})
    keys=list(pair_stats([0],[0],[0]))
    ds={k:sum(r[k] for r in detector) for k in keys};qs={k:sum(r[k] for r in pairs) for k in keys}
    result={'status':'INCOMPLETE_AGENT_PILOT' if missing else 'COMPLETE_AGENT_INPUTS','measurement_sha256':sha(mp),'qwen_sha256':sha(qp),
        'detector_scope':'Complete 9 matched condition pairs; all 441 timestamps answered by TalkNet threshold0.5. Not Qwen.',
        'detector_pairs':detector,'detector_totals':ds,'qwen_completed_sessions':len(q['runs']),'qwen_expected_sessions':36,'qwen_complete_pairs':pairs,'qwen_partial_totals':qs,'missing_sessions':missing,
        'investment_screen_passed':None if missing else 'Run report_benefit_screen.py for final evaluation',
        'reason':'Groq daily-token rate limit interrupted Qwen calls; no missing output treated as an error or a no-change outcome.',
        'decision':'Do not commit to full framework research on current evidence. The Qwen screen is incomplete, not a failed completed test.' if missing else 'Run full report.'}
    (P/'results/benefit_screen_progress.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Restoration benefit screen: completed and incomplete evidence','',
        '**Current recommendation: do not launch the full framework campaign on this evidence. This is not a completed negative Qwen hypothesis test.**','',
        'Three fixed two-second source clips (speaking, non-speaking, transition), with noise/FastDVDnet, reduced-resolution/FSRCNN, and JPEG/bilateral-filter pairs. Selection and the investment criterion were saved before inference.','',
        '## Complete local detector experiment','',
        f"All 9 pairs were executed. Captured TalkNet: {ds['captured_wrong']}/{ds['captured_answered']} wrong. Restored TalkNet: {ds['restored_wrong']}/{ds['restored_answered']} wrong. Corrections: {ds['corrections']}; introduced errors: {ds['new_errors']}.",'',
        f"A label-assisted timestamp selector has {ds['oracle_wrong_fixed_capture_coverage']}/{ds['captured_answered']} errors at identical captured coverage: no improvement. This exact finite-candidate upper bound applies to selecting the existing detector answers, not to a Qwen agent's potentially different reasoning or a newly trained predictor/fusion method.",'',
        '## Actual Qwen evidence','',
        f"Saved completed sessions: {len(q['runs'])}/36. Fully paired comparisons: {len(pairs)}/18. Missing sessions: {len(missing)}. Partial pairs show {qs['corrections']} corrections and {qs['new_errors']} introduced errors; they must not be generalized to missing clips or conditions.",'',
        'Groq returned HTTP 429 for the daily token quota (200,000 tokens/day). The runner is resumable and retains all completed responses. No smaller model, cached detector label, or fabricated prediction was substituted for the requested Qwen output.','',
        'The outstanding investment screen depends on the missing agent answers, including the transition clip. It is therefore unscored. Do not call it failed or claim all requested experiments finished.','',
        'The complete local bound supplies no evidence of useful detector-level corrections in this candidate set. That supports caution about investing in this particular framework, but does not disprove all restoration benefits, establish novelty, or settle the full Qwen question.','']
    (P/'BENEFIT_SCREEN_PROGRESS_2026-10-16.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'detector':ds,'qwen_completed':len(q['runs']),'paired':len(pairs),'missing':len(missing),'partial_qwen':qs},indent=2))

if __name__=='__main__':main()

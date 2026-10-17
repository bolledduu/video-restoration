"""Fixed-coverage upper bounds and complete benefit screen on actual Qwen outputs."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pair_stats(c,r,y):
    c,r,y=map(np.asarray,(c,r,y));assert c.shape==r.shape==y.shape
    ca,ra=c!=-1,r!=-1;cw=ca&(c!=y);rw=ra&(r!=y);common=ca&ra
    fixes=cw&ra&(r==y);harm=ca&(c==y)&rw
    oracle=c.copy();oracle[fixes]=r[fixes]
    assert np.array_equal(oracle!=-1,ca)
    assert int(np.sum(ca&(oracle!=y)))==int(cw.sum()-fixes.sum())
    return {'captured_wrong':int(cw.sum()),'captured_answered':int(ca.sum()),'restored_wrong':int(rw.sum()),'restored_answered':int(ra.sum()),
        'corrections':int(fixes.sum()),'new_errors':int(harm.sum()),'oracle_wrong_fixed_capture_coverage':int(cw.sum()-fixes.sum()),
        'common_answered':int(common.sum()),'captured_wrong_common':int((cw&common).sum()),'restored_wrong_common':int((rw&common).sum()),
        'clip_oracle_wrong_common':min(int((cw&common).sum()),int((rw&common).sum())),
        'new_correct_answers_after_capture_abstention':int(np.sum(~ca&ra&(r==y))),
        'lost_correct_answers_to_abstention':int(np.sum(ca&(c==y)&~ra))}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime-root',type=Path,default=P.parents[1]/'agent-audio-visual');root=parser.parse_args().runtime_root
    # Corrections require an answered wrong baseline, not a baseline abstention.
    t=pair_stats([0,1,-1,0],[1,0,1,-1],[1,1,1,0]);assert t['corrections']==1 and t['new_errors']==1 and t['oracle_wrong_fixed_capture_coverage']==0 and t['captured_answered']==3 and t['new_correct_answers_after_capture_abstention']==1
    source=P/'results/benefit_qwen.json';d=json.loads(source.read_text());assert d['status']=='COMPLETE' and len(d['runs'])==36
    assert sha(P/'scripts/benefit_qwen.py')==d['script_sha256'] and sha(P/'benefit_qwen_protocol.json')==d['protocol_sha256']
    assert sha(P/'results/benefit_measurements.json')==d['protocol']['measurement_sha256']
    assert len({r['configuration_sha256'] for r in d['runs']})==1
    rows=[];policies=[]
    for case in d['cases']:
        assert sha(case['path'])==case['sha256'] and sha(case['outputs'])==case['output_sha256']
        with np.load(case['path']) as a:
            all_labels=a['labels'].copy();y=all_labels[case['indices']].copy()
            assert hashlib.sha256(a['audio'].tobytes()).hexdigest()==case['audio_array_sha256']
        annotation=root/'data/unitalk'/(case['source']+'.csv');assert sha(annotation)==case['csv_sha256']
        annotations=[r for r in csv.reader(annotation.read_text().splitlines()) if r[7]==case['entity'] and case['start']-1e-6<=float(r[1])<=case['start']+1.96+1e-6]
        assert len(annotations)==50 and [int(r[8]) for r in annotations]==all_labels.tolist()
        assert all(r[6]==('SPEAKING_AUDIBLE' if int(r[8]) else 'NOT_SPEAKING') for r in annotations)
        assert case['candidates']['captured']['audio_probability']==case['candidates']['restored']['audio_probability']==case['candidates']['reference']['audio_probability']
        assert y.tolist()==case['scored_labels'] and len(y)==49
        for repeat in range(2):
            runs={name:next(r for r in d['runs'] if (r['case_id'],r['repeat'],r['condition'])==(case['id'],repeat,name)) for name in ['captured','restored']}
            for name,run in runs.items():
                assert run['status']=='completed' and all(r['model']==d['protocol']['model'] for r in run['responses'])
                tags=np.array(run['submission']['frame_tags'])
                assert run['evaluation']['frame_wrong']==int(np.sum((tags!=-1)&(tags!=y)))
                assert run['evaluation']['majority_correct']==(run['submission']['majority_speaking']==bool(np.mean(y)>.5))
                payloads=[json.loads(t['content']) for t in run['trace'] if t['role']=='tool' and 'audio_visual_speaking_scores' in t.get('content','')]
                assert payloads and all(t['audio_visual_speaking_scores']==case['candidates'][name]['probability'] for t in payloads)
            c,r=[runs[name]['submission'] for name in ['captured','restored']];cm,rm=[case['candidates'][name] for name in ['captured','restored']]
            row={'case_id':case['id'],'source':case['source'],'kind':case['kind'],'condition':case['condition'],'repeat':repeat,
                **pair_stats(c['frame_tags'],r['frame_tags'],y),'majority_captured_correct':runs['captured']['evaluation']['majority_correct'],'majority_restored_correct':runs['restored']['evaluation']['majority_correct'],
                'musiq_captured':cm['musiq'],'musiq_restored':rm['musiq'],'brisque_captured':cm['brisque'],'brisque_restored':rm['brisque']}
            rows.append(row)
            ct,rt=np.array(c['frame_tags']),np.array(r['frame_tags']);disagree=(np.array(cm['probability'])>=.5)!=(np.array(rm['probability'])>=.5)
            options={'captured_only':ct,'restored_only':rt,'disagreement_capture':np.where(disagree,ct,rt),
                'musiq_selection':rt if rm['musiq']>cm['musiq'] else ct,
                'brisque_selection':rt if rm['brisque']<cm['brisque'] else ct,
                'qwen_confidence_selection':rt if r['confidence']>c['confidence'] else ct}
            for name,tags in options.items():
                mask=(tags!=-1)&(ct!=-1)
                policies.append({'case_id':case['id'],'repeat':repeat,'policy':name,'wrong':int(np.sum((tags!=-1)&(tags!=y))),'answered':int(np.sum(tags!=-1)),
                    'common_with_capture':int(mask.sum()),'wrong_common':int(np.sum(mask&(tags!=y))),'captured_wrong_common':int(np.sum(mask&(ct!=y)))})
    totals=[]
    countkeys=list(pair_stats([0],[0],[0]))
    for repeat in range(2):
        subset=[r for r in rows if r['repeat']==repeat];t={k:sum(r[k] for r in subset) for k in countkeys}
        t['repeat']=repeat;t['oracle_gain_percentage_points']=100*t['corrections']/t['captured_answered'] if t['captured_answered'] else None;totals.append(t)
    positive=[];negative=[]
    for case in d['cases']:
        subset=[r for r in rows if r['case_id']==case['id']]
        if all(r['corrections']>0 for r in subset):positive.append(case['id'])
        if all(r['new_errors']>0 for r in subset):negative.append(case['id'])
    passed=bool(positive and negative and all(t['oracle_gain_percentage_points'] is not None and t['oracle_gain_percentage_points']>=5 for t in totals))
    summary={}
    for name in {r['policy'] for r in policies}:
        subset=[r for r in policies if r['policy']==name];summary[name]={k:sum(r[k] for r in subset) for k in ['wrong','answered','common_with_capture','wrong_common','captured_wrong_common']}
    report={'status':'COMPLETE','source_sha256':sha(source),'report_script_sha256':sha(__file__),'rows':rows,'totals_by_repeat':totals,'repeatable_benefit_cases':positive,'repeatable_harm_cases':negative,
        'investment_screen_passed':passed,'policy_summary':summary,'policy_rows':policies,'audit':{'sessions':36,'calls':sum(len(r['responses']) for r in d['runs']),'sources':3,'tracks':3,'conditions_per_track':3,'repetitions':2,'unique_source_timestamps':147,'matched_input_output_hashes':True,'scoring_and_payloads_checked':True},
        'limitations':['Oracle uses labels and is not deployable.','Same source frames repeated across degradations and runs, not independent samples.','Three purposively selected two-second tracks; category and source confounded.','One Qwen tool-agent; no raw-media Qwen inference.','JPEG filtering is a weak classical baseline; no exhaustive model search.','Synthetic defects on already imperfect source crops.','No novelty, full-framework, or paper-ready claim.']}
    transport_log=P/'results/benefit_transport_errors.jsonl'
    report['transport_exceptions']={'initial_http_400_body_not_retained':True,
        'recorded_errors':[json.loads(line) for line in transport_log.read_text().splitlines()] if transport_log.exists() else [],
        'policy':'At most two identical-request retries for provider tool_use_failed; quota retries retain request. Successful responses are conditional on completing the API protocol; retries are not independent repetitions.'}
    (P/'results/benefit_screen_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (P/'results/benefit_screen_pairs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Restoration benefit and oracle screen — 16 October 2026','',
        '**Investment screen: '+('PASS — proceed to focused method development and larger validation.' if passed else 'FAIL — do not launch the full restoration-selection framework campaign from this tested setup.')+'**','',
        'This decision applies to the candidate pool and small sample tested, not all restoration methods or the validity of the research questions. A pass would support research investment, not certify a novel method or accepted paper.','',
        '## Fixed experiment','',
        'Three two-second clips, selected by annotations before inference: speaking, listening, and a speaking-state transition. Three degradation/restoration pairs per clip; 36 fresh Qwen sessions in two counterbalanced repeats. The agent sees quality and perception-tool measurements, not raw video. Dataset labels and reference images are evaluation-only. The source/category assignment is confounded and cannot support population claims.','',
        '- Gaussian noise sigma25 → FastDVDnet with sigma25.',
        '- 4× reduced resolution → pretrained FSRCNN x4. Captured-only uses bicubic upsampling, not a deliberately weak nearest-neighbor baseline.',
        '- JPEG quality10 → bilateral filtering (fixed classical baseline, not a learned artifact-removal model).','',
        'FSRCNN weights: https://github.com/Saafke/FSRCNN_Tensorflow . Exact downloaded weight hash and all inputs/outputs are saved.','',
        'The screen fixed before inference requires repeatable corrections on at least one source, repeatable new errors on at least one source, and at least five percentage points of timestamp-oracle error reduction in both repeats at unchanged captured-answer coverage. This threshold is an operational investment criterion, not statistical significance.','',
        '## Paired results','',
        '| Source / category | Condition | Captured errors (r1/r2) | Restored errors (r1/r2) | Captured answered (r1/r2) | Restored answered (r1/r2) | Corrections (r1/r2) | New errors (r1/r2) |','|---|---|---:|---:|---:|---:|---:|---:|']
    for case in d['cases']:
        subset=[r for r in rows if r['case_id']==case['id']]
        def vals(k):return '/'.join(str(r[k]) for r in subset)
        lines.append(f"| {case['source']} / {case['kind']} | {case['condition']} | {vals('captured_wrong')} | {vals('restored_wrong')} | {vals('captured_answered')} | {vals('restored_answered')} | {vals('corrections')} | {vals('new_errors')} |")
    lines += ['', 'Each session has 49 scored timestamps. Errors exclude abstentions. Corrections require captured wrong → restored correct on the same answered timestamp. Correct answers after a captured abstention are recorded separately, never counted as corrections.','',
        '## Oracle: is there benefit worth preserving?','',
        '| Repeat | Captured errors | Captured answered | Timestamp-oracle errors at same coverage | Oracle reduction (percentage points) |','|---|---:|---:|---:|---:|']
    for t in totals:
        gain='N/A' if t['oracle_gain_percentage_points'] is None else f"{t['oracle_gain_percentage_points']:.2f}"
        lines.append(f"| {t['repeat']+1} | {t['captured_wrong']} | {t['captured_answered']} | {t['oracle_wrong_fixed_capture_coverage']} | {gain} |")
    lines += ['', f'Repeatable benefit cases: {positive}. Repeatable harmful cases: {negative}.','',
        'The timestamp oracle keeps the exact captured-answer mask and substitutes only known correct restored tags at captured errors. It cannot improve by abstaining more. This is a highly optimistic, label-assisted upper bound on selecting existing answers, not a real selector and not a bound on a newly trained predictor or fusion method. The separate whole-clip oracle on jointly answered timestamps is recorded in the audit JSON; it may have less headroom than timestamp-level selection. No human-quality preservation is implied by this task oracle.','',
        '## Deployable baseline rules','',
        '| Policy | Errors across both repeats | Answered /882 | Errors on common timestamps vs captured errors there |','|---|---:|---:|---|']
    for name,s in sorted(summary.items()):lines.append(f"| {name} | {s['wrong']} | {s['answered']} | {s['wrong_common']} vs {s['captured_wrong_common']} ({s['common_with_capture']} common) |")
    lines += ['', 'The 882 opportunities repeat 147 source timestamps across three degradations and two runs. They are not 882 independent observations. Compare errors with coverage; common-mask comparisons are not a full risk–coverage curve. No rule receives labels.','',
        '## Relation to the four RQs','',
        'RQ1 receives condition-specific paired attribution evidence; visual-quality measurements are in the measurement JSON. RQ2 still requires a real assessor and an independent evaluation of harm versus correction. RQ3 requires a deployable method to recover oracle headroom while preserving quality; an oracle is not that method. RQ4 has not been validated across agent models, datasets, or unseen restoration families. Both a positive and negative investment decision must respect these limits.','',
        '## Exact Qwen responses','']
    for r in d['runs']:
        lines += [f"### {r['case_id']} / {r['condition']} / repeat {r['repeat']+1}",'',f"Majority speaking: {r['submission']['majority_speaking']}; confidence: {r['submission']['confidence']}; evaluation: {r['evaluation']}",'',r['submission']['rationale'],'']
    lines += ['', '## Operational exceptions', '', 'Provider quota waits retained the in-flight request. An initial HTTP 400 body was not retained; subsequent provider errors are recorded in results/benefit_transport_errors.jsonl. Tool-generation failures received at most two identical-request retries. Reported predictions are conditional on successful protocol completion; failed calls and retries are not independent repetitions.', '']
    (P/'BENEFIT_ORACLE_DECISION_2026-10-16.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'audit':report['audit'],'passed':passed,'benefit':positive,'harm':negative,'totals':totals,'baselines':summary},indent=2))

if __name__=='__main__':main()

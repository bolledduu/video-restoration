"""Audit all selected transfer cases and report RQ-specific scope, not paper validation."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from evaluate_qwen_capture_guard import apply_guard,test_guard

P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def auc(labels,scores):
    pos=[s for y,s in zip(labels,scores) if y];neg=[s for y,s in zip(labels,scores) if not y]
    if not pos or not neg:return None
    return float(np.mean([int(a>b)+.5*int(a==b) for a in pos for b in neg]))

def main():
    test_guard()
    path=P/'results/transfer_rq_qwen.json';d=json.loads(path.read_text())
    assert d['status']=='COMPLETE' and len(d['runs'])==24
    assert sha(P/'scripts/transfer_rq_qwen.py')==d['script_sha256']
    assert sha(P/'transfer_rq_qwen_protocol.json')==d['protocol_sha256']
    assert sha(P/'results/transfer_rq_measurements.json')==d['protocol']['measurement_sha256']
    assert len({r['configuration_sha256'] for r in d['runs']})==1
    policies=['captured_only','restored_only','detector_disagreement_abstain','detector_disagreement_capture','captured_answer_priority']
    pairs=[];guard_rows=[]
    for c in d['cases']:
        assert sha(c['path'])==c['sha256'] and sha(c['outputs'])==c['output_sha256']
        with np.load(c['path']) as a:labels=a['labels'][:24].copy()
        assert np.all(labels==c['label'])
        for repeat in range(2):
            def get(condition):
                found=[r for r in d['runs'] if (r['case_id'],r['repeat'],r['condition'])==(c['id'],repeat,condition)]
                assert len(found)==1
                r=found[0];s=r['submission'];tags=np.array(s['frame_tags'])
                assert r['status']=='completed'
                assert r['evaluation']['frame_wrong']==int(np.sum((tags!=-1)&(tags!=labels)))
                assert r['evaluation']['majority_correct']==(s['majority_speaking']==bool(c['label']))
                assert all(x['model']==d['protocol']['model'] for x in r['responses'])
                evidence=[json.loads(t['content']) for t in r['trace'] if t['role']=='tool' and 'audio_visual_speaking_scores' in t.get('content','')]
                assert evidence and all(e['audio_visual_speaking_scores']==c['candidates'][condition]['probability'] for e in evidence)
                return s
            captured=get('captured');ct=np.array(captured['frame_tags']);cm=c['candidates']['captured']
            for restorer in ['fastdvdnet','spatial_nlm']:
                restored=get(restorer);rt=np.array(restored['frame_tags']);rm=c['candidates'][restorer];common=(ct!=-1)&(rt!=-1)
                wrongc=(ct!=-1)&(ct!=labels);wrongr=(rt!=-1)&(rt!=labels)
                new=int(np.sum(common & (ct==labels)&(rt!=labels)))
                fixes=int(np.sum(common & (ct!=labels)&(rt==labels)))
                disagreement=float(np.mean((np.array(cm['probability'])>=.5)!=(np.array(rm['probability'])>=.5)))
                pairs.append({'case_id':c['id'],'source':c['source'],'label':c['label'],'start':c['start'],'restorer':restorer,'repeat':repeat,
                    'captured_wrong':int(wrongc.sum()),'restored_wrong':int(wrongr.sum()),'captured_answered':int((ct!=-1).sum()),'restored_answered':int((rt!=-1).sum()),
                    'common_answered':int(common.sum()),'new_errors_common':new,'corrections_common':fixes,
                    'majority_correct_to_wrong':captured['majority_speaking']==bool(c['label']) and restored['majority_speaking']!=bool(c['label']),
                    'both_quality_improve':rm['musiq']>cm['musiq'] and rm['brisque']<cm['brisque'],
                    'musiq_delta':rm['musiq']-cm['musiq'],'brisque_delta':rm['brisque']-cm['brisque'],
                    'risk':{'low_musiq':-rm['musiq'],'high_brisque':rm['brisque'],'low_qwen_confidence':1-restored['confidence'],
                            'detector_disagreement':disagreement,'low_detector_confidence':float(np.mean(1-np.maximum(np.array(rm['probability']),1-np.array(rm['probability']))))}})
                for policy in policies:
                    answer=apply_guard(policy,captured,restored,cm['probability'],rm['probability']);tags=np.array(answer['frame_tags']);mask=(tags!=-1)&(ct!=-1)
                    guard_rows.append({'case_id':c['id'],'restorer':restorer,'repeat':repeat,'policy':policy,'frame_tags':answer['frame_tags'],
                        'wrong':int(np.sum((tags!=-1)&(tags!=labels))),'answered':int(np.sum(tags!=-1)),
                        'majority_wrong':int(answer['majority_speaking']!=-1 and answer['majority_speaking']!=c['label']),
                        'majority_answered':int(answer['majority_speaking']!=-1),
                        'common_with_capture':int(mask.sum()),'wrong_on_common':int(np.sum(mask&(tags!=labels))),
                        'captured_wrong_on_common':int(np.sum(mask&(ct!=labels)))})
    candidates=[]
    for c in d['cases']:
        for restorer in ['fastdvdnet','spatial_nlm']:
            rows=[r for r in pairs if r['case_id']==c['id'] and r['restorer']==restorer]
            candidates.append({'case_id':c['id'],'source':c['source'],'restorer':restorer,
                'common_answered_both_repeats':all(r['common_answered']>0 for r in rows),
                'repeated_harm':all(r['new_errors_common']>0 for r in rows),
                'repeated_correction':all(r['corrections_common']>0 for r in rows),
                'primary_harm':all(r['new_errors_common']>0 and r['both_quality_improve'] for r in rows),
                'risk':{k:float(np.mean([r['risk'][k] for r in rows])) for k in rows[0]['risk']}})
    risk_results={}
    risk_eligible=[c for c in candidates if c['common_answered_both_repeats']]
    exclusive=[c for c in risk_eligible if c['repeated_harm']!=c['repeated_correction']]
    for key in candidates[0]['risk']:
        risk_results[key]={'harm_vs_other_auc':auc([c['repeated_harm'] for c in risk_eligible],[c['risk'][key] for c in risk_eligible]),
            'harm_vs_correction_auc':auc([c['repeated_harm'] for c in exclusive],[c['risk'][key] for c in exclusive])}
    summary={}
    for restorer in ['fastdvdnet','spatial_nlm']:
        summary[restorer]={}
        for policy in policies:
            rows=[r for r in guard_rows if r['restorer']==restorer and r['policy']==policy]
            summary[restorer][policy]={k:sum(r[k] for r in rows) for k in ['wrong','answered','majority_wrong','majority_answered','common_with_capture','wrong_on_common','captured_wrong_on_common']}
    harm_sources=sorted({c['source'] for c in candidates if c['primary_harm']})
    corrected=sorted({c['case_id'] for c in candidates if c['repeated_correction']})
    screen=bool(harm_sources and corrected)
    result={'status':'COMPLETE','source_sha256':sha(path),'script_sha256':sha(__file__),'pairs':pairs,'candidates':candidates,'guards':guard_rows,'guard_summary':summary,
        'risk_auc':risk_results,'primary_harm_sources':harm_sources,'repeated_corrected_tracks':corrected,'focused_method_development_screen':screen,
        'audit':{'sessions':24,'api_calls':sum(len(r['responses']) for r in d['runs']),'sources':2,'tracks':4,'unique_scored_timestamps':96,'repeats':2,'reference_and_labels_excluded_from_agent':True},
        'scope':'Same Qwen, same dataset, one synthetic Gaussian noise severity. Two source videos not in earlier paired Qwen test; not untouched sources in entire project. No novel assessor/framework or cross-agent generalization established.'}
    (P/'results/transfer_rq_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with (P/'results/transfer_rq_pairs.csv').open('w',newline='') as f:
        flat=[{k:v for k,v in r.items() if k!='risk'} for r in pairs];w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=['# Four-RQ source-transfer pilot — 16 October 2026','',
        '**Focused method-development screen: '+('PASS' if screen else 'FAIL')+'.** This is not validation of all four RQs, novelty, or paper readiness.','',
        'Two additional source videos; the earliest eligible speaking and non-speaking track from each, selected before inference. Four tracks, real CPU FastDVDnet and spatial NLM outputs, fixed Gaussian noise sigma 25. Twenty-four actual Qwen tool-agent sessions: three evidence conditions × four tracks × two repeats. No severity or sample search after outcomes. Labels/reference images stay outside the agent tools.','',
        'The source videos are new to the paired Qwen experiment, but some tracks were already used in detector-only experiments. This is source transfer from the prior Qwen pilot, not an untouched external validation set. Qwen consumes detector/quality measurements, not pixels or waveforms.','',
        '## RQ1 — failure replication','',
        '| Track | Restorer | Captured wrong tags (r1/r2) | Restored wrong tags (r1/r2) | New errors on common answered timestamps | Corrections on common timestamps | Overall correct to wrong (r1/r2) | Both quality scores improve? |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for c in d['cases']:
        for restorer in ['fastdvdnet','spatial_nlm']:
            rows=[r for r in pairs if r['case_id']==c['id'] and r['restorer']==restorer]
            def vals(k):return '/'.join(str(r[k]) for r in rows)
            lines.append(f"| {c['source']}, label {c['label']}, {c['start']:.2f}s | {restorer} | {vals('captured_wrong')} | {vals('restored_wrong')} | {vals('new_errors_common')} | {vals('corrections_common')} | {vals('majority_correct_to_wrong')} | {rows[0]['both_quality_improve']} |")
    lines += ['',f'Repeatable primary-harm sources: {harm_sources}. Repeatable corrected tracks: {corrected}.', '',
        'Wrong-label counts exclude abstentions. A zero count does not establish correctness if no timestamps were answered; coverage is reported below. Repeats are not independent videos.', '',
        '| Track | Evidence | MUSIQ (higher better) | BRISQUE (lower better) | TalkNet errors /24 |',
        '|---|---|---:|---:|---:|']
    for c in d['cases']:
        for name,m in c['candidates'].items():
            lines.append(f"| {c['source']}, label {c['label']} | {name} | {m['musiq']:.3f} | {m['brisque']:.3f} | {m['wrong_tags']} |")
    lines += ['',
        'On the strongest repeated new regression (`qv3-HaaxGUc:6`, 14.84–15.84s), FastDVDnet changes Qwen errors from 6 to 24 with full coverage in both repeats. MUSIQ rises from 14.386 to 20.194; BRISQUE decreases only slightly, from 46.288 to 46.245. The latter is a numerical improvement, not evidence of a meaningful or statistically significant quality gain. Reference-video TalkNet also has 24/24 errors, so the result does not establish invented mouth details; restoration can remove noise that happened to help an already unreliable detector.', '',
        'Spatial NLM on that same track gives 15 errors/21 answered timestamps in repeat 1, but 0 errors/8 answered in repeat 2. Its overall answer is wrong in both. Do not present the smaller error count as improved accuracy: coverage and answers changed substantially even at temperature zero.', '',
        '## RQ2 — assessment baselines','',
        f'There are {len(candidates)} track/restorer candidates, of which {sum(c["repeated_harm"] for c in candidates)} have repeated new errors and {sum(c["repeated_correction"] for c in candidates)} have repeated corrections. Only {len(risk_eligible)} candidates have at least one jointly answered timestamp in both repeats and are eligible for the observed-error ranking analysis. No common answers is not evidence of preserved attribution. Risk scores are averaged over repeats; repeats are not treated as independent candidates.', '',
        '| Risk score | Harm vs other AUROC | Harm vs correction AUROC |','|---|---:|---:|']
    for name,values in risk_results.items():
        lines.append('| '+name+' | '+' | '.join('Not estimable (missing class)' if v is None else f'{v:.3f}' for v in values.values())+' |')
    lines += ['', 'Harm-vs-correction excludes candidates with neither outcome or both outcomes. AUC, where calculable, is descriptive on this tiny known-design sample; no confidence or superiority claim is supported. No new assessor was trained. These baselines cannot establish the RQ2 claim of performance beyond existing baselines.', '',
        '## RQ3 — frozen guard transfer','',
        '| Restorer | Policy | Wrong timestamp tags | Answered /192 | Wrong overall / answered |','|---|---|---:|---:|---:|']
    for restorer,pols in summary.items():
        for name,s in pols.items():
            lines.append(f"| {restorer} | {name} | {s['wrong']} | {s['answered']}/192 | {s['majority_wrong']}/{s['majority_answered']} |")
    lines += ['', 'Captured-only rows are the same eight Qwen sessions reused as the comparator for both restoration families. Do not sum them as independent observations. Guards use the exact previous fixed rules and no labels. They route answers, not image pixels. This does not measure perceptual quality of a novel image-selection framework. Common-answered comparisons are saved in the audit JSON; they restrict to timestamps answered by both systems and are not a full matched-coverage risk curve.', '',
        '## RQ4 — what transferred','',
        'The frozen guards and unchanged Qwen prompts were applied to both additional sources and two restoration families. This tests a limited transfer slice. Only one agent model, one dataset and one noise severity were run. Generalization across unseen agent models, datasets and degradation types remains untested; no framework exists to validate on all those axes.', '',
        '## Investment decision','',
        'Before inference, the screening rule required both repeatable quality-improving harm on a new source and a repeatable correction of a captured Qwen error on some track. This rule tests whether there is evidence for a useful restoration-versus-risk tradeoff. The earlier majority-answer screen remains unchanged; this timestamp screen is a separately declared follow-up.', '',
        ('The screen passes: the observed tradeoff motivates focused method development, but the baselines do not establish a novel architecture or final answers to all four RQs.' if screen else 'The screen fails: these additional cases do not justify launching the full framework experiment campaign under this fixed screen. This does not erase the earlier positive failure example and does not universally disprove the questions.'), '',
        'Both restorers are video-only. None of these results establishes audio-created visual details.','',
        '## Exact agent explanations','']
    for r in d['runs']:
        lines += [f"### {r['case_id']} / {r['condition']} / repeat {r['repeat']+1}",'',
            f"Majority speaking: {r['submission']['majority_speaking']}; confidence: {r['submission']['confidence']}; evaluation: {r['evaluation']}",'',r['submission']['rationale'],'']
    (P/'FOUR_RQ_TRANSFER_DECISION_2026-10-16.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'audit':result['audit'],'screen':screen,'harm_sources':harm_sources,'corrected_tracks':corrected,'guards':summary,'risk_auc':risk_results},indent=2))

if __name__=='__main__':main()

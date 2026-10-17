"""Deterministic post-answer guards on completed real Qwen calls.

This is a retrospective baseline evaluation on known cases, not new Qwen
inference, an independent confirmation set, or a trained novel method.
"""
import argparse
import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_guard(policy, captured, restored, captured_scores, restored_scores):
    """No labels, source identities, reference image, or quality labels accepted."""
    c = np.asarray(captured['frame_tags'], dtype=int)
    r = np.asarray(restored['frame_tags'], dtype=int)
    assert c.shape == r.shape == (24,)
    assert np.isin(c, [-1, 0, 1]).all() and np.isin(r, [-1, 0, 1]).all()
    disagree = (np.asarray(captured_scores) >= .5) != (np.asarray(restored_scores) >= .5)
    assert disagree.shape == (24,)
    if policy == 'captured_only':
        tags, majority = c.copy(), int(captured['majority_speaking'])
    elif policy == 'restored_only':
        tags, majority = r.copy(), int(restored['majority_speaking'])
    elif policy == 'detector_disagreement_abstain':
        tags = np.where(disagree, -1, r)
        majority = -1 if disagree.any() else int(restored['majority_speaking'])
    elif policy == 'detector_disagreement_capture':
        tags = np.where(disagree, c, r)
        majority = int(captured['majority_speaking'] if disagree.any() else restored['majority_speaking'])
    elif policy == 'captured_answer_priority':
        tags = np.where(c != -1, c, r)
        majority = int(captured['majority_speaking'])
    else:
        raise ValueError(policy)
    return {'frame_tags': tags.tolist(), 'majority_speaking': majority,
            'detector_disagreement_indices': np.flatnonzero(disagree).tolist()}


def test_guard():
    c = {'frame_tags': [0] * 24, 'majority_speaking': False}
    r = {'frame_tags': [1] * 24, 'majority_speaking': True}
    # Conflict must use the captured answer even if an external label would
    # make the restored answer correct: the policy cannot consult labels.
    assert apply_guard('detector_disagreement_capture', c, r, [.1]*24, [.9]*24)['frame_tags'] == [0]*24
    assert apply_guard('detector_disagreement_abstain', c, r, [.1]*24, [.9]*24)['frame_tags'] == [-1]*24
    assert apply_guard('detector_disagreement_capture', c, r, [.1]*24, [.2]*24)['frame_tags'] == [1]*24
    unknown = {'frame_tags': [-1]*24, 'majority_speaking': False}
    assert apply_guard('detector_disagreement_capture', unknown, r, [.1]*24, [.9]*24)['frame_tags'] == [-1]*24
    assert apply_guard('captured_answer_priority', unknown, r, [.1]*24, [.9]*24)['frame_tags'] == [1]*24


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.runtime_root.resolve()
    p = Path(__file__).resolve().parents[1]
    test_guard()
    source = p / 'results/three_video_qwen_paired.json'
    d = json.loads(source.read_text())
    assert d['status'] == 'COMPLETE' and len(d['runs']) == 20
    assert sha(p/'scripts/three_video_qwen_paired.py') == d['script_sha256']
    assert sha(p/'three_video_qwen_protocol.json') == d['protocol_sha256']
    assert len({r['configuration_sha256'] for r in d['runs']}) == 1
    policies = ['captured_only', 'restored_only', 'detector_disagreement_abstain',
                'detector_disagreement_capture', 'captured_answer_priority']
    protocol = {
        'scope': __doc__, 'source_sha256': sha(source), 'script_sha256': sha(__file__),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'policies': policies, 'detector_threshold': .5,
        'primary': 'Timestamp errors and answered timestamp count, with captured-only as the essential baseline.',
        'success_for_guard_utility': 'Fewer errors than restored-only without fewer answered timestamps.',
        'not_sufficient_for_method_research': 'Matching captured-only is not incremental task benefit or novelty.',
        'independence': 'Same known five tracks, two repeats; no held-out validation, no new API calls, no tuned thresholds.',
        'audio_claim': 'Check actual restorer interface and runtime calls; do not mistake downstream audio fusion for audio-conditioned pixel generation.'}
    # Written before guard outcome calculation, but explicitly retrospective.
    (p/'qwen_capture_guard_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows = []
    for case in d['cases']:
        cid = case['id']
        extra = cid == '9Zja_cIAwY8'
        bundle = root/'data/quick_temporal'/f'{cid}.npz' if extra else root/'data/agent_pilot'/f"{cid.rsplit('__sigma',1)[0]}.npz"
        arrays = p/'data/additional_fastdvdnet_videos'/f'{cid}.npz' if extra else root/'data/agent_pilot'/f'{cid}_outputs.npz'
        assert sha(bundle) == case['bundle_sha256'] and sha(arrays) == case['arrays_sha256']
        annotation = root/'data/unitalk'/f"{case['source']}.csv"
        assert sha(annotation) == case['annotation_sha256']
        with np.load(bundle) as data:
            labels = data['labels'][:24].copy()
            assert sha_bytes(data['audio'].tobytes()) == case['audio_sha256']
        assert len(labels) == 24 and np.all(labels == case['label'])
        for repeat in range(2):
            def get(condition):
                candidates = [r for r in d['runs'] if (r['case_id'],r['repeat'],r['condition']) == (cid,repeat,condition)]
                assert len(candidates) == 1 and candidates[0]['status'] == 'completed'
                return candidates[0]['submission']
            captured, restored = get('captured'), get('restored')
            cp, rp = [case['candidates'][name]['probability'] for name in ['captured','fastdvdnet']]
            outputs = {policy: apply_guard(policy, captured, restored, cp, rp) for policy in policies}
            for policy, answer in outputs.items():
                tags = np.asarray(answer['frame_tags'])
                captured_tags = np.asarray(captured['frame_tags'])
                restored_tags = np.asarray(restored['frame_tags'])
                wrong = (tags != -1) & (tags != labels)
                captured_correct = (captured_tags != -1) & (captured_tags == labels)
                captured_wrong = (captured_tags != -1) & (captured_tags != labels)
                common = (tags != -1) & (restored_tags != -1)
                rows.append({'case_id':cid, 'source':case['source'], 'repeat':repeat, 'policy':policy,
                    **answer, 'wrong_tags':int(wrong.sum()), 'answered_tags':int((tags!=-1).sum()),
                    'majority_wrong':int(answer['majority_speaking']!=-1 and answer['majority_speaking']!=case['label']),
                    'majority_answered':int(answer['majority_speaking']!=-1),
                    'new_errors_vs_captured':int((captured_correct & wrong).sum()),
                    'corrections_vs_captured':int((captured_wrong & (tags==labels)).sum()),
                    'common_answered_with_restored':int(common.sum()),
                    'wrong_on_common_guard':int((wrong & common).sum()),
                    'wrong_on_common_restored':int(((restored_tags!=labels)&common).sum())})
    summary = {}
    keys = ['wrong_tags','answered_tags','majority_wrong','majority_answered','new_errors_vs_captured',
            'corrections_vs_captured','common_answered_with_restored','wrong_on_common_guard','wrong_on_common_restored']
    for policy in policies:
        subset = [r for r in rows if r['policy']==policy]
        summary[policy] = {key:sum(r[key] for r in subset) for key in keys}
        summary[policy]['timestamp_coverage'] = summary[policy]['answered_tags']/240
        summary[policy]['selective_error_rate'] = summary[policy]['wrong_tags']/summary[policy]['answered_tags']
    # Audit actual inference interface and both saved generation entry points.
    files = [root/'external/fastdvdnet/fastdvdnet.py', root/'external/fastdvdnet/models.py',
             root/'scripts/agent_pilot_measure.py', p/'scripts/check_additional_fastdvdnet_videos.py']
    signatures, calls = [], []
    for file in files:
        tree = ast.parse(file.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'denoise_seq_fastdvdnet':
                signatures.append({'file':str(file),'line':node.lineno,'args':[a.arg for a in node.args.args]})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'denoise_seq_fastdvdnet':
                calls.append({'file':str(file),'line':node.lineno,'arguments':[ast.unparse(a) for a in node.args]})
    assert signatures and signatures[0]['args'] == ['seq','noise_std','temp_psz','model_temporal']
    assert len(calls) >= 2
    mechanism = {'finding':'FastDVDnet has no audio input; this pipeline cannot test audio-created visual hallucinations.',
                 'interpretation':'Audio is used later by TalkNet. That is audiovisual inference, not audio-conditioned restoration.',
                 'signatures':signatures,'actual_calls':calls,'files_sha256':{str(f):sha(f) for f in files}}
    report = {'status':'COMPLETE','protocol':protocol,'source_sha256':sha(source),'rows':rows,'summary':summary,
              'mechanism_audit':mechanism,'checks':'Hashes, label arrays, uniform base configuration, guard unit checks and actual restorer call signatures verified.',
              'limitations':'Known cases; no independent validation; two repeats are not two videos; no novelty claim; post-answer gating, not fresh Qwen reasoning.'}
    (p/'results/qwen_capture_guard.json').write_text(json.dumps(report, indent=2)+'\n')
    with (p/'results/qwen_capture_guard_summary.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['policy']+list(summary[policies[0]])); w.writeheader()
        w.writerows({'policy':name,**values} for name,values in summary.items())
    lines = ['# Capture guard evaluation — 16 October 2026','',
        '**Decision: NO to a full-paper campaign for the original audio-created visual hallucination claim or this simple guard as a novel method.**', '',
        'A narrower restoration-induced tool-agent error is demonstrated. The guard repairs that known example, but does not beat retaining captured evidence. This is a completed retrospective baseline test, not an independent validation study.', '',
        '## What was tested','',
        'Five tracks from three videos, two repetitions each. The inputs are the 20 completed real Qwen responses from the fixed pilot. This script makes no fresh model calls: it implements an automatic post-answer guard. All rules receive only the model answers and detector scores; evaluation labels and reference images are unavailable to the guard.', '',
        '- Detector-disagreement abstention: return unknown where captured/restored detector decisions differ; abstain on the overall answer if any timestamp differs.',
        '- Detector-disagreement capture: use the captured Qwen tag at each disagreement; keep restored tags elsewhere. Use the captured overall answer if any detector timestamp differs.',
        '- Captured-answer priority: use captured tags whenever answered, otherwise restored tags; use the captured overall answer. This is a conservative baseline, not a novel proposal.', '',
        'All detector decisions use the existing 0.5 threshold. No training or threshold tuning was performed. Rules were specified after the failures were known, so this cannot be represented as prospective validation.', '',
        '## Results','',
        '| System | Wrong timestamp labels | Answered timestamps /240 | Coverage | Wrong overall answers / answered |',
        '|---|---:|---:|---:|---:|']
    for name,s in summary.items():
        lines.append(f"| {name} | {s['wrong_tags']} | {s['answered_tags']}/240 | {s['timestamp_coverage']:.1%} | {s['majority_wrong']}/{s['majority_answered']} |")
    lines += ['', 'The 240 timestamp opportunities are 120 unique timestamps evaluated twice, not 240 independent examples. No confidence interval or population success rate is inferred.', '',
        '## Concrete failure and repair','',
        'On `EEey6Pt0rqI:2`, 7.28–8.28 seconds, dataset labels mark the visible person as NOT_SPEAKING. The captured Qwen answer has 0/24 wrong timestamp tags. The restored answer has 5/24 wrong tags (0.00–0.16 seconds relative to the clip). The capture guard replaces those five tags with captured answers and reaches 0/24 errors while answering all 24 timestamps, in both repeats. It does not identify a specific word or prove fabricated mouth motion.', '',
        'The repair is real on these saved responses. However, captured-only already has zero errors and answers more timestamps across the pilot than the detector-disagreement capture guard. Captured-answer priority matches the captured-only baseline here. Therefore this evaluation establishes no incremental accuracy/coverage benefit over retaining the captured input and no novel contribution.', '',
        '## Audio-created visual details: mechanism check','', mechanism['finding'], '', mechanism['interpretation'], '',
        'The restorer receives video frames and a noise level. Its callable interface and actual generation calls were inspected and recorded with source hashes. Changing audio is not an input intervention to this restorer. These results neither demonstrate nor disprove the hypothesis for a genuinely audio-conditioned restoration model.', '',
        '## Research decision','',
        '- Narrow failure motivation: supported by the specific timestamp counterexample.',
        '- Repair of the known failure: supported for the deterministic capture guard.',
        '- Better method than captured-only: not demonstrated; the tested guard is worse in coverage at the same zero observed error count.',
        '- Audio-created visual hallucinations: not tested by this architecture; reject this claim for the current results.',
        '- Full-paper investment on the current method/claim: NO. Do not scale this baseline merely to obtain a positive result.', '',
        'The scientifically defensible next contribution would need to preserve a genuine benefit of restoration while avoiding these errors, beyond simply keeping captured evidence. That is a future falsifiable requirement, not a result of this pilot. The earlier failed majority-answer replication criterion remains failed and is not replaced by this post hoc guard result.', '']
    (p/'QWEN_CAPTURE_GUARD_DECISION_2026-10-16.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','summary':summary,'mechanism':mechanism['finding']}, indent=2))


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


if __name__ == '__main__':
    main()

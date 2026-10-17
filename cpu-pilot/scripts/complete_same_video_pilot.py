"""Complete a bounded same-video selection pilot, preserving the original gate.

Only missing FastDVDnet quality scores are newly inferred. Restorations and
TalkNet outputs come from the audited prior runs; this is not new replication.
No tuning, agent API calls, human-quality claims, or novel-method claims.
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime-root', required=True, type=Path)
    args = ap.parse_args()
    root = args.runtime_root.resolve()
    package = Path(__file__).resolve().parents[1]
    result_path = package / 'results/same_video_complete_pilot.json'
    paths = {n: package / 'results' / n for n in [
        'agent_pilot_measurements.json', 'quick_temporal_pilot.json',
        'additional_fastdvdnet_videos.json', 'restoration_before_after_reproduction.json']}
    old, transfer, extra, replay = [json.loads(p.read_text()) for p in paths.values()]
    protocol = {
        'scope': 'Same-video CPU selection and baseline assessment pilot, not full causal/method validation',
        'primary_gate': old['protocol']['primary_screen'],
        'primary_cases': 'Original 8 cases, original 4 candidates; gate unchanged',
        'expanded_cases': 'All 8 original and 3 additional cases; common captured/spatial_nlm/fastdvdnet candidates',
        'exclusions': 'Unverified AV-SpeakerBench labels excluded from accuracy; DMD gaze not an ASD test',
        'quality': 'BRISQUE/MUSIQ means over frames 0,12,24; SSIM diagnostic only',
        'selection_rules': ['retain', 'always_nlm', 'always_fastdvdnet', 'musiq', 'brisque', 'confidence',
                            'agreement_guard', 'disagreement_guard_10pct', 'disagreement_guard_zero', 'oracle'],
        'detection': 'Fixed, untrained risk scores on noninput candidates: disagreement, uncertainty, input MSE, negative MUSIQ, BRISQUE',
        'harm_label': 'At least one correct-to-incorrect frame; report net-worse cases separately',
        'coverage': 'Full coverage for every selection rule; no abstention',
        'sensitivity': 'Repeat summaries on tracks whose source-reference predictions are all correct',
        'limits': 'Reused sources; 3 additional tracks all silent; no held-out trained assessor or enhancer transfer',
    }
    protocol_path = package / 'same_video_complete_protocol.json'
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING',
              'protocol': protocol, 'protocol_sha256': sha(protocol_path), 'script_sha256': sha(__file__),
              'input_sha256': {n: sha(p) for n, p in paths.items()}, 'new_quality': []}
    result_path.write_text(json.dumps(report, indent=2) + '\n')
    print('Protocol saved; computing nine missing quality-frame scores', flush=True)
    for folder in ['tools/quality_cpu', 'tools/iqa_models', 'tools/python']:
        sys.path.insert(0, str(root / folder))
    import cv2
    import numpy as np
    import torch
    import pyiqa
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.manual_seed(0)
    metric = pyiqa.create_metric('musiq', device='cpu').eval()
    bd = root / 'external/brisque'
    brisque = cv2.quality.QualityBRISQUE_create(str(bd / 'brisque_model_live.yml'), str(bd / 'brisque_range_live.yml'))
    quality_hashes = {n: sha(bd / n) for n in ['brisque_model_live.yml', 'brisque_range_live.yml']}
    quality_hashes['musiq'] = sha(Path(torch.hub.get_dir()) / 'pyiqa/musiq_koniq_ckpt-e95806b9.pth')
    assert quality_hashes == transfer['quality_hashes']
    assert quality_hashes['musiq'] == old['weights']['musiq']
    assert extra['weights']['fastdvdnet'] == old['weights']['fastdvdnet']
    report['quality_hashes'] = quality_hashes
    cases = []
    for c in old['cases']:
        path = root / 'data/agent_pilot' / (c['id'] + '_outputs.npz')
        assert sha(path) == c['output_sha256']
        with np.load(path) as a:
            candidates = {}
            for name in ['captured', 'spatial_nlm', 'fastdvdnet']:
                m = dict(c['candidates'][name])
                mse = float(np.mean((a[name].astype(float) - a['captured'].astype(float)) ** 2))
                assert abs(mse - m['input_mse']) < 1e-8
                candidates[name] = m
        cases.append({'id': c['id'], 'source': c['source'], 'label': c['label'], 'sigma': c['sigma'],
                      'candidates': candidates, 'reference': c['candidates']['reference']})
    for c in extra['cases']:
        sid = c['id']
        t = next(t for t in transfer['cases'] if t['source'] == sid)
        path = package / 'data/additional_fastdvdnet_videos' / (sid + '.npz')
        oldpath = root / 'data/quick_temporal' / (sid + '_outputs.npz')
        assert sha(path) == c['output_sha256'] and sha(oldpath) == t['output_sha256']
        with np.load(path) as a, np.load(oldpath) as b:
            assert np.array_equal(a['captured'], b['degraded'])
            candidates = {'captured': dict(t['candidates']['degraded']),
                          'spatial_nlm': dict(t['candidates']['spatial_nlm']),
                          'fastdvdnet': dict(c['candidates']['fastdvdnet'])}
            bs, ms = [], []
            for i in [0, 12, 24]:
                video = a['fastdvdnet']
                bs.append(float(brisque.compute(cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR))[0]))
                with torch.inference_mode():
                    ms.append(float(metric(torch.from_numpy(video[i].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255).item()))
            candidates['fastdvdnet'].update(brisque=float(np.mean(bs)), musiq=float(np.mean(ms)))
            for name, m in candidates.items():
                im = b['spatial_nlm'] if name == 'spatial_nlm' else a[name]
                m['input_mse'] = float(np.mean((im.astype(float) - a['captured'].astype(float)) ** 2))
                probs = np.asarray(m['probability'])
                m['confidence'] = float(np.mean(np.maximum(probs, 1 - probs)))
            report['new_quality'].append({'source': sid, 'brisque_frames': bs, 'musiq_frames': ms,
                                          'output_sha256': sha(path)})
        cases.append({'id': sid + '__sigma25', 'source': sid, 'label': c['label'], 'sigma': 25,
                      'candidates': candidates, 'reference': c['candidates']['reference']})
        result_path.write_text(json.dumps(report, indent=2) + '\n')
        print(sid, 'quality measured', flush=True)

    def enrich(case):
        captured = np.asarray(case['candidates']['captured']['probability']) >= .5
        correct = captured == case['label']
        case['reference_all_correct'] = bool(np.all((np.asarray(case['reference']['probability']) >= .5) == case['label']))
        for m in case['candidates'].values():
            p = np.asarray(m['probability'])
            assert len(p) == 24 and np.isfinite(p).all()
            candidate_correct = (p >= .5) == case['label']
            m.update(wrong_frames=int((~candidate_correct).sum()),
                     new_errors=int((correct & ~candidate_correct).sum()),
                     corrections=int((~correct & candidate_correct).sum()),
                     disagreement=float(np.mean((p >= .5) != captured)))

    for case in cases:
        enrich(case)
        c = case['candidates']
        win = {m: min(c, key=lambda n: (sign * c[n][m], n)) for m, sign in [('musiq', -1), ('brisque', 1), ('confidence', -1)]}
        w = win['musiq']
        case['choices'] = dict(retain='captured', always_nlm='spatial_nlm', always_fastdvdnet='fastdvdnet', **win,
            agreement_guard=w if w == win['brisque'] else 'captured',
            disagreement_guard_10pct='captured' if c[w]['disagreement'] > .1 else w,
            disagreement_guard_zero='captured' if c[w]['disagreement'] > 0 else w,
            oracle=min(c, key=lambda n: (c[n]['wrong_frames'], -c[n]['musiq'], n)))
        case['harmful_consensus'] = win['musiq'] == win['brisque'] and c[w]['wrong_frames'] > c['captured']['wrong_frames']

    def summarize(selected):
        out = {}
        for rule in cases[0]['choices']:
            records = [c['candidates'][c['choices'][rule]] for c in selected]
            n = 24 * len(selected)
            per_source = {}
            for source in sorted({c['source'] for c in selected}):
                subset = [c for c in selected if c['source'] == source]
                per_source[source] = sum(c['candidates'][c['choices'][rule]]['wrong_frames'] for c in subset) / (24 * len(subset))
            out[rule] = {'frames': n, 'coverage': 1, 'wrong_frames': sum(m['wrong_frames'] for m in records),
                'error_rate': sum(m['wrong_frames'] for m in records) / n,
                'new_errors': sum(m['new_errors'] for m in records), 'corrections': sum(m['corrections'] for m in records),
                'mean_musiq': float(np.mean([m['musiq'] for m in records])),
                'mean_brisque': float(np.mean([m['brisque'] for m in records])),
                'mean_ssim': float(np.mean([m['ssim'] for m in records])),
                'per_source_error': per_source, 'source_balanced_error': float(np.mean(list(per_source.values())))}
        return out

    def auc(y, score):
        positive, negative = [s for s, t in zip(score, y) if t], [s for s, t in zip(score, y) if not t]
        if not positive or not negative:
            return None
        return sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))

    def detection(selected):
        records = [m for c in selected for n, m in c['candidates'].items() if n != 'captured']
        y = [m['new_errors'] > 0 for m in records]
        funcs = {'disagreement': lambda m: m['disagreement'], 'uncertainty': lambda m: 1 - m['confidence'],
                 'input_mse': lambda m: m['input_mse'], 'low_musiq': lambda m: -m['musiq'], 'high_brisque': lambda m: m['brisque']}
        return {'candidates': len(records), 'harmful_candidates': sum(y),
                'auroc': {k: auc(y, [f(m) for m in records]) for k, f in funcs.items()},
                'interpretation': 'Descriptive ranking of fixed baselines on reused cases; not validated novel assessor performance'}

    # Recompute the historical four-candidate gate rather than changing its pool.
    original_harm = sorted({c['source'] for c in old['cases'] if
        c['winners']['musiq'] == c['winners']['brisque'] and
        c['candidates'][c['winners']['musiq']]['error'] > c['candidates']['captured']['error'] + 1e-9})
    original_benefit = sorted({c['source'] for c in old['cases'] if
        min(c['candidates'][n]['error'] for n in old['protocol']['candidates'][1:]) < c['candidates']['captured']['error'] - 1e-9})
    assert original_harm == old['summary']['agreed_harmed_sources']
    report['primary_gate'] = {'harmful_sources': original_harm, 'beneficial_sources': original_benefit,
                              'pass': len(original_harm) >= 2 and len(original_benefit) >= 1}
    report['cases'] = cases
    report['selection'] = summarize(cases)
    clean = [c for c in cases if c['reference_all_correct']]
    report['reference_correct_sensitivity'] = {'cases': [c['id'] for c in clean], 'selection': summarize(clean), 'detection': detection(clean)}
    report['detection'] = detection(cases)
    report['expanded_harmful_consensus_sources'] = sorted({c['source'] for c in cases if c['harmful_consensus']})
    report['scope'] = {'sources': len({c['source'] for c in cases}), 'cases': len(cases), 'frames': 24 * len(cases),
                       'speaking_cases': sum(c['label'] == 1 for c in cases), 'silent_cases': sum(c['label'] == 0 for c in cases)}
    assert report['scope'] == {'sources': 5, 'cases': 11, 'frames': 264, 'speaking_cases': 4, 'silent_cases': 7}
    assert report['selection']['disagreement_guard_zero']['wrong_frames'] == report['selection']['retain']['wrong_frames']
    assert report['selection']['disagreement_guard_zero']['new_errors'] == 0
    for m in report['selection'].values():
        assert m['wrong_frames'] == report['selection']['retain']['wrong_frames'] + m['new_errors'] - m['corrections']
    report['decision'] = 'DO_NOT_COMMIT_TO_FULL_PAPER_ON_CURRENT_EVIDENCE' if not report['primary_gate']['pass'] else 'PILOT_GATE_PASSED_ONLY'
    report['status'] = 'COMPLETE'
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    result_path.write_text(json.dumps(report, indent=2) + '\n')
    with result_path.with_suffix('.csv').open('w', newline='') as s:
        rows = [dict(rule=k, **{a: b for a, b in v.items() if a != 'per_source_error'}) for k, v in report['selection'].items()]
        w = csv.DictWriter(s, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({k: report[k] for k in ['scope', 'primary_gate', 'detection', 'decision']}, indent=2), flush=True)
    for k, m in report['selection'].items():
        print(k, 'errors', m['wrong_frames'], 'new', m['new_errors'], 'corrected', m['corrections'], 'MUSIQ', round(m['mean_musiq'], 3), flush=True)


if __name__ == '__main__':
    main()

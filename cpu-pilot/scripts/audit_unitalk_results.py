"""Independent standard-library arithmetic audit of saved model scores."""
import hashlib
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def main():
    raw = json.loads((ROOT / 'results' / 'unitalk_cpu_frame_scores.json').read_text())
    report = json.loads((ROOT / 'results' / 'unitalk_cpu_evaluation.json').read_text())
    if len(raw) != 16 or len(report['cases']) != 16:
        raise ValueError('Incomplete results')
    flat = {'V': [], 'AV': []}
    for audit in raw:
        case = next(c for c in report['cases'] if c['id'] == audit['id'])
        y = audit['labels']
        if len(y) != case['frames_scored'] or not all(v == case['label'] for v in y):
            raise ValueError('Label/count mismatch')
        predictions = audit['predictions']
        for condition, heads in predictions.items():
            if heads['A'] != predictions['reference']['A']:
                raise ValueError('Audio-only scores changed')
            for head in ('V', 'AV'):
                scores = heads[head]
                if len(scores) != len(y) or any(not 0 <= s <= 1 for s in scores):
                    raise ValueError('Invalid scores')
                wrong = [score if label == 0 else 1 - score for label, score in zip(y, scores)]
                metrics = case['metrics'][condition][head]
                error = sum((score >= .5) != label for label, score in zip(y, scores)) / len(y)
                if not math.isclose(error, metrics['error_rate'], abs_tol=1e-12):
                    raise ValueError('Error rate arithmetic mismatch')
                if not math.isclose(sum(wrong) / len(y), metrics['mean_wrong_class_probability'], abs_tol=1e-6):
                    raise ValueError('Wrong-score arithmetic mismatch')
                if sum(s >= .9 for s in wrong) != metrics['wrong_score_at_least_0_9_frames']:
                    raise ValueError('High wrong-score count mismatch')
        for head in ('V', 'AV'):
            events = []
            for i, label in enumerate(y):
                frame = audit['frame_indices'][i]
                r, d, m, s = [predictions[k][head][i] for k in ('reference', 'degraded', 'matching', 'swapped')]
                delta = f32(s - m) if label == 0 else f32(f32(1 - s) - f32(1 - m))
                if ((r >= .5) == label and (d >= .5) == label and (m >= .5) == label
                    and (s >= .5) != label and delta >= .10
                    and audit['quality']['swapped']['ssim'][frame] > audit['quality']['degraded']['ssim'][frame]):
                    events.append(frame)
                identity = f"{audit['id']}:{frame}"
                tie = int(hashlib.sha256(identity.encode()).hexdigest()[:12], 16)
                flat[head].append(dict(label=label, p=s, tie=tie, confidence=-abs(f32(s-.5)),
                    conditioning_sensitivity=abs(f32(s-m)), capture_disagreement=abs(f32(s-d))))
            if events != case['joint_events'][head]['source_frame_indices']:
                raise ValueError('Joint event mismatch')
    for head, rows in flat.items():
        for coverage in (.5, .75):
            for rule in ('confidence', 'conditioning_sensitivity', 'capture_disagreement'):
                selected = sorted(rows, key=lambda r: (r[rule], r['tie']))[:int(len(rows)*coverage)]
                rate = sum((r['p'] >= .5) != r['label'] for r in selected) / len(selected)
                measured = report['safeguards'][head]['pooled'][str(coverage)][rule]
                if not math.isclose(rate, measured['error_rate'], abs_tol=1e-12):
                    raise ValueError('Selective-error arithmetic mismatch')
    sources = {c['source'] for c in report['cases'] if c['joint_events']['AV']['count'] >= 5}
    if (len(sources) >= 2) != report['RQ1_pass']:
        raise ValueError('RQ1 gate mismatch')
    expected = 'GO_BOUNDED_FOLLOWUP' if report['RQ1_pass'] and report['RQ3_pass'] else 'NO_GO_SCALE_TESTED_SETUP'
    if expected != report['decision']:
        raise ValueError('Decision mismatch')
    result = dict(status='PASSED', cases=len(raw), metrics='Errors, wrong scores, high-score counts, A invariance, joint events, three selection rankings, decision logic',
                  note='Separate standard-library calculation; does not re-run neural networks or independently re-annotate ground truth.')
    (ROOT / 'results' / 'unitalk_result_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Finish the laptop feasibility decision from saved real-model outputs.

Prediction disagreement is a diagnostic, never accuracy or false attribution.
No network, model downloads, GPU allocation or new inference is required.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADS = ('visual_only', 'audio_only', 'fixed_audio_visual')


def main():
    paths = [ROOT / 'results' / name for name in
             ('video_baseline.json', 'talknet_diagnostics.json')]
    baseline, predictions = [json.loads(p.read_text()) for p in paths]
    assert len(baseline) == 6 and len(predictions) == 18
    groups = {}
    for record in predictions:
        key = (record['sample_id'], record['level'])
        group = groups.setdefault(key, {})
        assert record['condition'] not in group
        group[record['condition']] = record
    assert len(groups) == 6
    rows = []
    for (sample, level), group in sorted(groups.items()):
        assert set(group) == {'reference', 'degraded', 'restored'}
        ref, deg, res = [group[c] for c in ('reference', 'degraded', 'restored')]
        n = ref['frame_count']
        for record in (deg, res):
            assert record['frame_count'] == n
            for field in ('timestamps', 'crop_boxes_256x144', 'audio_sha256', 'logits_audio_only'):
                assert record[field] == ref[field], (sample, level, field)
        for head in HEADS:
            logits = [r['logits_' + head] for r in (ref, deg, res)]
            assert all(len(values) == n for values in logits)
            assert all(len(pair) == 2 and all(math.isfinite(x) for x in pair)
                       for values in logits for pair in values)
            # Matches numpy.argmax: ties choose class 0. Raw labels are uncalibrated.
            labels = [[int(pair[1] > pair[0]) for pair in values] for values in logits]
            rows.append(dict(sample_id=sample, level=level, head=head, frames=n,
                reference_class1_frames=sum(labels[0]), degraded_class1_frames=sum(labels[1]),
                restored_class1_frames=sum(labels[2]),
                reference_vs_degraded_disagreements=sum(a != b for a, b in zip(labels[0], labels[1])),
                reference_vs_restored_disagreements=sum(a != b for a, b in zip(labels[0], labels[2])),
                degraded_vs_restored_disagreements=sum(a != b for a, b in zip(labels[1], labels[2]))))
    with (ROOT / 'results/cpu_paired_diagnostics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        investment_decision='NO', decision_scope='pursue a full paper from the current pilot',
        feasibility_assessment='CLOSED', scientific_hypothesis='UNTESTED',
        compute_policy='laptop only; no paid allocation; no further pilot preparation scheduled',
        decision_basis='Required positive evidence is absent; the executable audio-conditioned model was not available in the recorded search. This is not evidence that the effect is absent.',
        source_clips=3, dependent_quality_cases=6, talknet_condition_runs=18,
        psnr_improved_cases=sum(r['output_quality']['psnr_db'] > r['input_quality']['psnr_db'] for r in baseline),
        ssim_improved_cases=sum(r['output_quality']['ssim_rgb_mean'] > r['input_quality']['ssim_rgb_mean'] for r in baseline),
        changed_prediction_cases={h: sum(r['degraded_vs_restored_disagreements'] > 0 for r in rows if r['head'] == h) for h in HEADS},
        checks='All six groups have identical frame counts, timestamps, crops, downstream audio hashes and audio-only logits across conditions; all logits finite.',
        interpretation='Frame disagreements show prediction sensitivity to video-only processing. They are neither errors nor verified speaking states; reference predictions are not ground truth.',
        speaker_accuracy=None, delta_false_attribution=None,
        rq1='Not demonstrated; do not claim false corroboration.',
        rq2='Not demonstrated; do not claim non-identifiability.',
        rq3='Do not develop a new method from these results.',
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (ROOT / 'results/final_cpu_decision.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

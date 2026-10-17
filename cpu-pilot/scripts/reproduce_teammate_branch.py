"""Audit and replay the committed restoration experiment without altering it.

The staging directory needs read-only-in-use `external` and `tools` junctions
to the existing runtime checkout. Model files are read; no downloads/API calls.
The branch's DMD Qwen records have no paired restored condition and are audited,
not represented as a reproduced restoration experiment.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True, type=Path)
    parser.add_argument('--revision', default='5f8307aee5ffb0f83c7f17ee343a23bd9626d4a3')
    args = parser.parse_args()
    runtime = args.runtime_root.resolve()
    package = Path(__file__).resolve().parents[1]
    stage = package / 'data' / ('teammate_reproduction_' + args.revision[:7])
    for part in ['scripts', 'results', 'branch_results', 'data/raw', 'data/processed']:
        (stage / part).mkdir(parents=True, exist_ok=True)
    for part in ['external', 'tools']:
        if not (stage / part).exists():
            raise RuntimeError(f'Create a directory junction {stage / part} to {runtime / part}')

    def git_file(path):
        return subprocess.check_output(['git', '-C', str(runtime), 'show', f'{args.revision}:{path}'])

    audit = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'revision': args.revision,
        'run_type': 'same-source restoration and downstream inference replay; no tuning',
        'status': 'RUNNING',
        'script_sha256': digest(__file__),
        'branch_file_sha256': {},
        'inputs': [],
        'dmd_replay_status': 'NOT_EXECUTED: media absent locally; no paired restored DMD outputs committed',
        'ground_truth_verified_for_av_speakerbench': False,
    }
    scripts = ['cpu_media.py', 'video_baseline.py', 'talknet_cpu.py',
               'false_restoration_event.py', 'qwen_api_batch.py', 'qwen_api_probe.py']
    saved = ['source_downloads.json', 'cpu_degradations.json', 'video_baseline.json',
             'talknet_diagnostics.json', 'cpu_paired_diagnostics.csv',
             'qwen_groq_batch.json', 'qwen_groq_probe.json']
    for name in scripts:
        path = stage / 'scripts' / name
        path.write_bytes(git_file('scripts/' + name))
        audit['branch_file_sha256']['scripts/' + name] = digest(path)
    for name in saved:
        path = stage / 'branch_results' / name
        path.write_bytes(git_file('results/' + name))
        audit['branch_file_sha256']['results/' + name] = digest(path)

    for entry in json.loads((stage / 'branch_results/source_downloads.json').read_text()):
        source = runtime / 'data/raw' / (entry['sample_id'] + '.mp4')
        actual = digest(source)
        assert actual == entry['sha256'], (source, 'source differs from committed SHA-256')
        shutil.copy2(source, stage / 'data/raw' / source.name)
        audit['inputs'].append({'path': str(source), 'sha256': actual, 'matches_branch': True})
    for path in sorted((runtime / 'data/processed').glob('*.mp4')):
        shutil.copy2(path, stage / 'data/processed' / path.name)
        audit['inputs'].append({'path': str(path), 'sha256': digest(path),
                                'comparison': 'decoded input metrics checked against branch below'})
    assert len(list((stage / 'data/raw').glob('*.mp4'))) == 3
    assert len(list((stage / 'data/processed').glob('*.mp4'))) == 6

    output = package / 'results/teammate_branch_reproduction.json'
    output.write_text(json.dumps(audit, indent=2) + '\n')
    sys.path.insert(0, str(stage / 'scripts'))
    print('Re-running unmodified branch restoration code on three matching source hashes', flush=True)
    baseline = load_module('video_baseline', stage / 'scripts/video_baseline.py')
    baseline.main()
    print('Re-running unmodified branch TalkNet code on all 18 conditions', flush=True)
    detector = load_module('talknet_cpu', stage / 'scripts/talknet_cpu.py')
    detector.main()

    import numpy as np
    import torch
    import cv2
    import av
    audit['versions'] = {'python': sys.version, 'torch': torch.__version__,
                         'cv2': cv2.__version__, 'av': av.__version__}
    prior_quality = json.loads((stage / 'branch_results/video_baseline.json').read_text())
    new_quality = json.loads((stage / 'results/video_baseline.json').read_text())
    quality_by_key = {(r['sample_id'], r['level']): r for r in prior_quality}
    audit['quality_comparison'] = []
    for row in new_quality:
        previous = quality_by_key[(row['sample_id'], row['level'])]
        assert row['checkpoint_sha256'] == previous['checkpoint_sha256']
        assert row['source_frame_timestamps'] == previous['source_frame_timestamps']
        differences = {}
        for condition in ['input_quality', 'output_quality']:
            for metric in ['psnr_db', 'ssim_rgb_mean']:
                delta = row[condition][metric] - previous[condition][metric]
                differences[condition + '_' + metric] = delta
                assert abs(delta) < 1e-5, (row['sample_id'], condition, metric, delta)
        audit['quality_comparison'].append({'sample_id': row['sample_id'], 'level': row['level'],
                                            'deltas_vs_branch': differences})

    old_records = json.loads((stage / 'branch_results/talknet_diagnostics.json').read_text())
    new_records = json.loads((stage / 'results/talknet_diagnostics.json').read_text())
    old_map = {(r['sample_id'], r['level'], r['condition']): r for r in old_records}
    new_map = {(r['sample_id'], r['level'], r['condition']): r for r in new_records}
    assert len(old_map) == len(new_map) == 18
    audit['logit_comparison'] = []
    for key, row in new_map.items():
        previous = old_map[key]
        for field in ['audio_sha256', 'checkpoint_sha256', 'crop_boxes_256x144', 'frame_count', 'timestamps']:
            assert row[field] == previous[field], (key, field)
        for head in ['visual_only', 'audio_only', 'fixed_audio_visual']:
            before, after = [np.asarray(r['logits_' + head]) for r in [previous, row]]
            assert np.isfinite(after).all()
            changed = int(np.count_nonzero(before.argmax(1) != after.argmax(1)))
            assert changed == 0, (key, head, 'branch decisions not reproduced')
            audit['logit_comparison'].append({'sample_id': key[0], 'level': key[1],
                'condition': key[2], 'head': head, 'decision_differences': changed,
                'max_absolute_logit_difference': float(np.max(np.abs(after - before)))})

    rows = []
    for sample, level in sorted({key[:2] for key in new_map}):
        group = [new_map[(sample, level, c)] for c in ['reference', 'degraded', 'restored']]
        assert len({r['audio_sha256'] for r in group}) == 1
        assert group[0]['logits_audio_only'] == group[1]['logits_audio_only'] == group[2]['logits_audio_only']
        for head in ['visual_only', 'audio_only', 'fixed_audio_visual']:
            ref, deg, res = [np.asarray(r['logits_' + head]).argmax(1) for r in group]
            rows.append({'sample_id': sample, 'level': level, 'head': head, 'frames': len(ref),
                'reference_class1_frames': int(ref.sum()), 'degraded_class1_frames': int(deg.sum()),
                'restored_class1_frames': int(res.sum()),
                'reference_vs_degraded_disagreements': int(np.count_nonzero(ref != deg)),
                'reference_vs_restored_disagreements': int(np.count_nonzero(ref != res)),
                'degraded_vs_restored_disagreements': int(np.count_nonzero(deg != res)),
                'no_to_yes': int(np.count_nonzero((deg == 0) & (res == 1))),
                'yes_to_no': int(np.count_nonzero((deg == 1) & (res == 0)))})
    branch_table = list(csv.DictReader(io.StringIO((stage / 'branch_results/cpu_paired_diagnostics.csv').read_text())))
    assert len(branch_table) == len(rows)
    for previous, row in zip(branch_table, rows):
        for key, value in previous.items():
            assert str(row[key]) == value, (key, row, previous)

    # These are tests of the existing counter's semantics, not experiment events.
    counter = load_module('branch_event_counter', stage / 'scripts/false_restoration_event.py')
    demo = [{'clip_id': 'audit_fixture', 'frame': i, 'original': 0,
             'restored': int(i == 0), 'ground_truth': 1} for i in range(10)]
    audit['counter_semantics_audit'] = {
        'fixture_description': '10 complete synthetic pairs; one 0-to-1 change; true label 1 (a correction)',
        'branch_output': counter.compare_conditions(demo),
        'correct_prediction_flip_rate': 0.1,
        'correct_false_positive_count': 0,
        'issues': ['event_rate divides events by events rather than complete pairs',
                   'no ground-truth check: a correction is named a false-positive event',
                   'rows lacking clip/frame keys are silently skipped; zero is not a completed test'],
    }
    batch = json.loads((stage / 'branch_results/qwen_groq_batch.json').read_text())
    probe = json.loads((stage / 'branch_results/qwen_groq_probe.json').read_text())
    audit['qwen_record_audit'] = {
        'batch_records': len(batch), 'probe_records': len(probe),
        'unique_video_sha256': sorted({r['video_sha256'] for r in batch + probe}),
        'paired_restoration_records': 0,
        'batch_prompt': 'Look at this face image. Is the driver likely looking forward? Reply in one short sentence with yes/no and a confidence score.',
        'batch_sampled_images_per_video': 1,
        'batch_audio_sent': False,
        'counter_result_if_batch_passed_directly': counter.compare_conditions(batch),
        'interpretation': 'Two source gaze probes, not original/restored speaker comparisons; no accuracy label or calibrated confidence.',
    }
    audit['rows'] = rows
    audit['summary'] = {
        'source_videos': 3, 'restoration_runs': 6, 'talknet_runs': 18,
        'branch_decisions_reproduced': True,
        'audio_only_invariant': True,
        'paired_av_frames': sum(r['frames'] for r in rows if r['head'] == 'fixed_audio_visual'),
        'changed_av_frames': sum(r['degraded_vs_restored_disagreements'] for r in rows if r['head'] == 'fixed_audio_visual'),
        'changed_av_cases': sum(r['degraded_vs_restored_disagreements'] > 0 for r in rows if r['head'] == 'fixed_audio_visual'),
        'verified_false_attribution_events': None,
    }
    audit['status'] = 'PASS_REPRODUCED_AV_SPEAKERBENCH; DMD_NOT_A_PAIRED_RESTORATION_TEST'
    audit['finished_utc'] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(audit, indent=2) + '\n')
    csv_path = output.with_suffix('.csv')
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(audit['summary'], indent=2), flush=True)
    print('Saved audit: ' + str(output), flush=True)


if __name__ == '__main__':
    main()

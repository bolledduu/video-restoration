"""Score published speaker labels after a controlled, framewise audio intervention."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from evaluate_causal import TalkNetCPU, speaker_metrics
from prepare_causal import ROOT, sha, canonical_sha, array_sha


def image_metrics(reference, candidate):
    mse = np.mean((reference.astype(np.float64) - candidate) ** 2, axis=(1, 2, 3))
    return dict(psnr=[float(10 * np.log10(255 ** 2 / v)) if v else None for v in mse],
        ssim=[float(structural_similarity(a, b, channel_axis=2, data_range=255)) for a, b in zip(reference, candidate)])


def event_mask(y, reference, degraded, matching, swapped, ssim_swapped, ssim_degraded):
    wrong_matching = np.where(y == 0, matching, 1 - matching)
    wrong_swapped = np.where(y == 0, swapped, 1 - swapped)
    return (((reference >= .5) == y) & ((degraded >= .5) == y) & ((matching >= .5) == y)
            & ((swapped >= .5) != y) & (wrong_swapped - wrong_matching >= .10)
            & (ssim_swapped > ssim_degraded))


def selective(records, coverage):
    y = np.array([r['label'] for r in records])
    probability = np.array([r['p'] for r in records])
    n = max(1, int(len(records) * coverage))
    tie = np.array([int(hashlib.sha256(r['id'].encode()).hexdigest()[:12], 16) for r in records])
    result = {}
    for rule in ('conditioning_sensitivity', 'capture_disagreement', 'reprojection_residual', 'confidence'):
        scores = np.array([r[rule] for r in records])
        order = np.lexsort((tie, scores))[:n]
        result[rule] = dict(coverage=n / len(y), retained_positive=int(np.sum(y[order] == 1)),
                            retained_negative=int(np.sum(y[order] == 0)), **speaker_metrics(y[order], probability[order]))
    result['random_expected_error'] = float(np.mean((probability >= .5) != y))
    return result


def main():
    directory = ROOT / 'data' / 'unitalk'
    manifest = json.loads((directory / 'manifest.json').read_text())
    p = manifest['protocol']
    if p != json.loads((ROOT / 'unitalk_cpu_protocol.json').read_text()) or canonical_sha(p) != manifest['protocol_sha256']:
        raise ValueError('Protocol mismatch')
    execution_records = []
    for row in manifest['records']:
        if sha(directory / row['path']) != row['bundle_sha256']:
            raise ValueError('Input changed')
        for level in p['effective_resolutions']:
            name = f"{row['id']}__r{level}"
            target, receipt = directory / 'outputs' / (name + '.npz'), directory / 'outputs' / (name + '.json')
            record = json.loads(receipt.read_text())
            donor = next(r for r in manifest['records'] if r['id'] == row['donor_id'])
            expected = dict(id=name, protocol_sha256=manifest['protocol_sha256'], bundle_sha256=row['bundle_sha256'],
                donor_bundle_sha256=donor['bundle_sha256'], original_downstream_audio_sha256=row['original_audio_sha256'],
                source_audio_sha256=row['conditioning_sha256'], donor_audio_sha256=donor['conditioning_sha256'],
                output_sha256=sha(target), repeat_equal=True, frame_count=p['frames'], resolution=level,
                weights_sha256='fc99bed110e02f0507fa84c6043f759def33d681f22aef4dcdc14a3e1cc5bda5')
            if any(record.get(k) != v for k, v in expected.items()):
                raise ValueError('Restoration provenance mismatch: ' + name)
            execution_records.append(record)
    model = TalkNetCPU()
    cases, audits, pooled = [], [], {'V': [], 'AV': []}
    for row in manifest['records']:
        with np.load(directory / row['path'], allow_pickle=False) as data:
            reference, source_audio, labels = data['reference'], data['original_audio'], data['labels']
        if array_sha(source_audio) != row['original_audio_sha256'] or not np.all(labels == row['label']):
            raise ValueError('Audio/label mismatch')
        boxes = [[0, 0, 128, 128]] * len(reference)
        for level in p['effective_resolutions']:
            name = f"{row['id']}__r{level}"
            with np.load(directory / 'outputs' / (name + '.npz'), allow_pickle=False) as data:
                videos = {k: data[k] for k in ('degraded', 'visual_only', 'matching', 'swapped')}
                small = data['small']
            videos['reference'] = reference
            if any(v.shape != reference.shape or v.dtype != np.uint8 for v in videos.values()):
                raise ValueError('Invalid output geometry/type')
            quality = {k: image_metrics(reference, video) for k, video in videos.items() if k != 'reference'}
            predictions = {}
            grid = None
            for condition, video in videos.items():
                predictions[condition], indices = model.predict(video, source_audio, boxes, 25)
                if grid is not None and not np.array_equal(grid, indices):
                    raise ValueError('Downstream frame grids differ')
                grid = indices
                if any(not np.isfinite(v).all() for v in predictions[condition].values()):
                    raise ValueError('Invalid model score')
            if any(not np.array_equal(v['A'], predictions['reference']['A']) for v in predictions.values()):
                raise ValueError('Downstream audio evidence changed')
            y = labels[grid]
            metrics = {condition: {head: speaker_metrics(y, v[head]) for head in ('V', 'AV')} for condition, v in predictions.items()}
            for condition, values in predictions.items():
                for head in ('V', 'AV'):
                    wrong_score = np.where(y == 0, values[head], 1 - values[head])
                    metrics[condition][head]['wrong_score_at_least_0_9_frames'] = int(np.sum(wrong_score >= .9))
                    metrics[condition][head]['brier_score'] = float(np.mean((values[head] - y) ** 2))
            events = {}
            for head in ('V', 'AV'):
                probs = {k: v[head] for k, v in predictions.items()}
                mask = event_mask(y, probs['reference'], probs['degraded'], probs['matching'], probs['swapped'],
                    np.asarray(quality['swapped']['ssim'])[grid], np.asarray(quality['degraded']['ssim'])[grid])
                events[head] = dict(count=int(mask.sum()), source_frame_indices=grid[mask].tolist(),
                    passes_track_screen=bool(mask.sum() >= 5),
                    audio_caused_correct_to_wrong=int(np.sum(((probs['matching'] >= .5) == y) & ((probs['swapped'] >= .5) != y))),
                    audio_caused_wrong_to_correct=int(np.sum(((probs['matching'] >= .5) != y) & ((probs['swapped'] >= .5) == y))),
                    swapped_quality_better_than_degraded_frames=int(np.sum(np.asarray(quality['swapped']['ssim'])[grid] > np.asarray(quality['degraded']['ssim'])[grid])),
                    wrong_class_score_delta_swap_minus_matching=float(np.mean(np.where(y == 0, probs['swapped'] - probs['matching'], probs['matching'] - probs['swapped']))))
                for j, index in enumerate(grid):
                    reproduced = np.asarray(Image.fromarray(videos['swapped'][index]).resize((8, 8), Image.Resampling.BICUBIC))
                    residual = float(np.mean(np.abs(reproduced.astype(float) - small[index])))
                    pooled[head].append(dict(id=f'{name}:{index}', source=row['source'], resolution=level,
                        label=int(y[j]), p=float(probs['swapped'][j]),
                        conditioning_sensitivity=float(abs(probs['swapped'][j] - probs['matching'][j])),
                        capture_disagreement=float(abs(probs['swapped'][j] - probs['degraded'][j])),
                        reprojection_residual=residual, confidence=float(-abs(probs['swapped'][j] - .5))))
            case = dict(id=name, source=row['source'], label=row['label'], resolution=level, frames_scored=len(grid),
                metrics=metrics, quality={k: {m: float(np.mean(values)) for m, values in v.items()} for k, v in quality.items()},
                joint_events=events, pixel_audio_mae=float(np.abs(videos['matching'].astype(float) - videos['swapped']).mean()),
                audio_only_invariant=True)
            cases.append(case)
            # Preserve complete frame-level audit, without exporting faces/audio.
            audit = dict(id=name,
                frame_indices=grid.tolist(), labels=y.tolist(), predictions={k: {h: a.tolist() for h, a in v.items()} for k, v in predictions.items()},
                quality=quality)
            audits.append(audit)
            (directory / 'outputs' / (name + '_scores.json')).write_text(json.dumps(audit, indent=2) + '\n')
            print(name, 'events', {h: events[h]['count'] for h in events}, 'AV errors', {k: round(v['AV']['error_rate'], 3) for k, v in metrics.items()}, flush=True)
    sources = sorted(set(row['source'] for row in manifest['records']))
    safeguards = {}
    for head, records in pooled.items():
        safeguards[head] = dict(pooled={str(c): selective(records, c) for c in (.5, .75)},
            by_source={s: {str(c): selective([r for r in records if r['source'] == s], c) for c in (.5, .75)} for s in sources},
            by_resolution={str(level): {str(c): selective([r for r in records if r['resolution'] == level], c) for c in (.5, .75)} for level in p['effective_resolutions']},
            abstain_all=dict(coverage=0, error_rate=None))
    evidence_sources = sorted({c['source'] for c in cases if c['joint_events']['AV']['passes_track_screen']})
    rq1 = len(evidence_sources) >= 2
    safeguards_pass = {}
    for method in ('conditioning_sensitivity', 'capture_disagreement', 'reprojection_residual'):
        pool = safeguards['AV']['pooled']
        better_sources = [s for s, v in safeguards['AV']['by_source'].items()
                          if v['0.75'][method]['error_rate'] < v['0.75']['confidence']['error_rate']]
        safeguards_pass[method] = dict(passes=all(pool[c][method]['error_rate'] < pool[c]['confidence']['error_rate'] for c in ('0.5', '0.75')) and len(better_sources) >= 2,
                                      improved_sources_at_75=better_sources)
    rq3 = any(v['passes'] for v in safeguards_pass.values())
    report = dict(status='COMPLETED_LABELED_CPU_PILOT', decision='GO_BOUNDED_FOLLOWUP' if rq1 and rq3 else 'NO_GO_SCALE_TESTED_SETUP',
        protocol_sha256=manifest['protocol_sha256'], sources=len(sources), clips=len(manifest['records']), conditions_per_degradation=3,
        degradations=len(p['effective_resolutions']), restoration_forward_passes=len(cases) * (3 * p['frames'] + 1),
        talknet_checkpoint_sha256=model.checkpoint_sha256, RQ1_pass=rq1, RQ1_evidence_sources=evidence_sources,
        RQ3_pass=rq3, safeguard_screen=safeguards_pass, cases=cases, safeguards=safeguards,
        interpretation='Measured screen for this framewise restorer and published labels. Not a universal hypothesis verdict, temporal-restorer benchmark, or calibrated confidence claim. Frames, degradation variants, and paired tracks are not independent sources.')
    report['execution_records'] = execution_records
    report['completed_combination_wall_seconds'] = sum(r['wall_seconds'] for r in execution_records)
    report['implementation_sha256'] = {name: sha(ROOT / 'scripts' / name) for name in
        ('ear_cpu_probe.py', 'run_unitalk_cpu.py', 'evaluate_unitalk_cpu.py', 'evaluate_causal.py', 'prepare_unitalk_cpu.py')}
    (ROOT / 'results' / 'unitalk_cpu_frame_scores.json').write_text(json.dumps(audits, indent=2, allow_nan=False) + '\n')
    (ROOT / 'results' / 'unitalk_cpu_evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'safeguards', 'execution_records')}, indent=2))


if __name__ == '__main__':
    main()

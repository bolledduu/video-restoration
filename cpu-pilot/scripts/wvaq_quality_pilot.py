"""Frozen, retrospective blind-IQA selection experiment; no training or label-based tuning."""
import hashlib
import itertools
import json
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'quality_cpu'))
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sharpen(video, amount):
    return np.stack([np.clip(np.rint(x.astype(np.float32) + amount *
        (x.astype(np.float32) - cv2.GaussianBlur(x.astype(np.float32), (0, 0), 1))),
        0, 255).astype(np.uint8) for x in video])


def choose(candidates, available, rule):
    # Only the explicitly marked oracle rules access reference/label outcomes.
    field, sign = {'brisque': ('brisque', 1), 'confidence': ('confidence', -1),
                   'ssim_oracle': ('ssim', -1), 'error_oracle': ('error', 1)}[rule]
    return min(available, key=lambda name: (sign * candidates[name][field], name))


def selection(candidates, pool, budget, rule):
    if not 1 <= budget <= len(pool) or len(set(pool)) != len(pool):
        raise ValueError('Invalid candidate pool/budget')
    winners = [choose(candidates, subset, rule) for subset in itertools.combinations(pool, budget)]
    fields = ('error', 'visual_error', 'brier', 'brisque', 'ssim', 'psnr', 'confidence')
    return {**{f: float(np.mean([candidates[w][f] for w in winners])) for f in fields},
            'choice_counts': {w: winners.count(w) for w in sorted(set(winners))}, 'subsets': len(winners)}


def aggregate(cases, pool, budget, rule):
    summaries = [selection(c['candidates'], pool, budget, rule) for c in cases]
    fields = ('error', 'visual_error', 'brier', 'brisque', 'ssim', 'psnr', 'confidence')
    return {f: float(np.mean([s[f] for s in summaries])) for f in fields}


def analyze(cases, protocol):
    pools = {name: protocol[key] for name, key in [('primary', 'primary_pool'),
        ('conventional', 'conventional_control_pool'), ('duplicate', 'duplicate_control_pool')]}
    groups = {'all': cases}
    groups.update({f'source:{s}': [c for c in cases if c['source'] == s] for s in sorted({c['source'] for c in cases})})
    groups.update({f'resolution:{r}': [c for c in cases if c['resolution'] == r] for r in (8, 4)})
    curves = {g: {p: {rule: {str(k): aggregate(rows, pool, k, rule) for k in protocol['budgets']}
                      for rule in ('brisque', 'confidence', 'ssim_oracle', 'error_oracle')}
                  for p, pool in pools.items()} for g, rows in groups.items()}
    primary = curves['all']['primary']['brisque']
    delta = primary['4']['error'] - primary['1']['error']
    gain = primary['1']['brisque'] - primary['4']['brisque']
    positive_sources = [g.removeprefix('source:') for g, v in curves.items() if g.startswith('source:')
        and v['primary']['brisque']['4']['error'] > v['primary']['brisque']['1']['error'] + 1e-12]
    screen = protocol['screen']
    conditions = dict(error_increase_at_least_threshold=delta >= screen['minimum_absolute_error_increase'],
        brisque_improvement_at_least_threshold=gain >= screen['minimum_mean_brisque_decrease'],
        enough_sources=len(positive_sources) >= screen['minimum_sources_with_error_increase'],
        both_resolutions=all(curves[f'resolution:{r}']['primary']['brisque']['4']['error'] >
                             curves[f'resolution:{r}']['primary']['brisque']['1']['error'] for r in (8, 4)))
    return dict(curves=curves, screen_checks=conditions,
        screen_pass=all(conditions.values()), error_increase=delta, brisque_improvement=gain,
        sources_with_error_increase=positive_sources,
        degraded_only={f: float(np.mean([c['candidates']['degraded'][f] for c in cases]))
                       for f in ('error', 'visual_error', 'brier', 'brisque', 'ssim', 'psnr')})


def main():
    from evaluate_causal import TalkNetCPU
    cv2.setNumThreads(2)
    protocol_path = ROOT / 'wvaq_quality_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    local = ROOT / 'data' / 'wvaq_quality'
    local.mkdir(exist_ok=True)
    model_dir = ROOT / 'external' / 'brisque'
    model_dir.mkdir(exist_ok=True)
    model_files = {}
    for name in ('brisque_model_live.yml', 'brisque_range_live.yml'):
        path = model_dir / name
        url = 'https://raw.githubusercontent.com/opencv/opencv_contrib/4.12.0/modules/quality/samples/' + name
        if not path.exists():
            with urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        model_files[name] = dict(url=url, sha256=sha(path))
    brisque = cv2.quality.QualityBRISQUE_create(str(model_dir / 'brisque_model_live.yml'), str(model_dir / 'brisque_range_live.yml'))
    base = ROOT / 'data' / 'unitalk'
    manifest = json.loads((base / 'manifest.json').read_text())
    prior_path = ROOT / 'results' / 'unitalk_cpu_frame_scores.json'
    prior = {r['id']: r for r in json.loads(prior_path.read_text())}
    detector = TalkNetCPU()
    prior_eval = json.loads((ROOT / 'results' / 'unitalk_cpu_evaluation.json').read_text())
    if detector.checkpoint_sha256 != prior_eval['talknet_checkpoint_sha256']:
        raise ValueError('Downstream checkpoint changed')
    cases = []
    for row in manifest['records']:
        if sha(base / row['path']) != row['bundle_sha256']:
            raise ValueError('Changed source bundle')
        with np.load(base / row['path'], allow_pickle=False) as bundle:
            ref, audio, labels = bundle['reference'], bundle['original_audio'], bundle['labels']
        for resolution in (8, 4):
            name = f"{row['id']}__r{resolution}"
            output = base / 'outputs' / (name + '.npz')
            receipt = json.loads((base / 'outputs' / (name + '.json')).read_text())
            if sha(output) != receipt['output_sha256'] or not receipt['repeat_equal']:
                raise ValueError('Changed restoration')
            old = prior[name]
            frame_indices = np.array(old['frame_indices'])
            y = labels[frame_indices]
            if y.tolist() != old['labels']:
                raise ValueError('Changed labels')
            with np.load(output, allow_pickle=False) as arrays:
                videos = {k: arrays[k] for k in ('degraded', 'visual_only', 'matching')}
            for amount in (.5, 1, 2):
                videos[f'unsharp_{amount}'] = sharpen(videos['degraded'], amount)
            candidates = {}
            for condition, video in videos.items():
                if condition in old['predictions']:
                    scores = {h: np.array(a) for h, a in old['predictions'][condition].items()}
                else:
                    scores, grid = detector.predict(video, audio, [[0, 0, 128, 128]] * len(video), 25)
                    if not np.array_equal(grid, frame_indices):
                        raise ValueError('Downstream time grid changed')
                if not np.array_equal(scores['A'], old['predictions']['reference']['A']):
                    raise ValueError('Audio-only scores changed')
                q = [float(brisque.compute(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[0]) for frame in video]
                ssim = [float(structural_similarity(a, b, channel_axis=2, data_range=255)) for a, b in zip(ref, video)]
                mse = np.mean((ref.astype(float) - video.astype(float)) ** 2, axis=(1, 2, 3))
                if np.any(mse <= 0):
                    raise ValueError('Unexpected identical reference candidate')
                psnr = (10 * np.log10(255 ** 2 / mse)).tolist()
                if not np.isfinite(q + ssim + psnr).all() or any(not np.isfinite(v).all() for v in scores.values()):
                    raise ValueError('Nonfinite metric')
                candidates[condition] = dict(brisque=float(np.mean(q)), ssim=float(np.mean(ssim)),
                    psnr=float(np.mean(psnr)), error=float(np.mean((scores['AV'] >= .5) != y)),
                    visual_error=float(np.mean((scores['V'] >= .5) != y)),
                    brier=float(np.mean((scores['AV'] - y) ** 2)), confidence=float(np.mean(np.maximum(scores['AV'], 1 - scores['AV']))),
                    frame_brisque=q, frame_ssim=ssim, frame_psnr=psnr,
                    predictions={h: v.tolist() for h, v in scores.items()},
                    pixel_sha256=hashlib.sha256(video.tobytes()).hexdigest())
            for i in (1, 2, 3):
                candidates[f'duplicate_{i}'] = candidates['degraded']
            case = dict(id=name, source=row['source'], resolution=resolution, labels=y.tolist(),
                        frame_indices=frame_indices.tolist(), restoration_sha256=sha(output), candidates=candidates)
            cases.append(case)
            (local / (name + '.json')).write_text(json.dumps(case, indent=2) + '\n')
            print(name, 'quality and downstream inference complete', flush=True)
    analysis = analyze(cases, protocol)
    report = dict(protocol=protocol, protocol_sha256=sha(protocol_path), cases=cases, **analysis,
        provenance=dict(opencv=cv2.__version__, numpy=np.__version__, brisque_models=model_files,
            downstream_checkpoint_sha256=detector.checkpoint_sha256, script_sha256=sha(__file__),
            previous_scores_sha256=sha(prior_path), source_manifest_sha256=sha(base / 'manifest.json')),
        counts=dict(sources=4, clips=8, clip_resolution_cases=len(cases), new_speaker_inferences=len(cases)*3,
                    new_brisque_frame_evaluations=len(cases)*6*25, cached_restoration_candidates=len(cases)*2),
        status='COMPLETED_RETROSPECTIVE_FEASIBILITY',
        decision='PASS_FOR_INDEPENDENT_VALIDATION' if analysis['screen_pass'] else 'REJECT_TESTED_FAILURE_HYPOTHESIS')
    (ROOT / 'results' / 'wvaq_quality_pilot.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('decision', 'screen_checks', 'error_increase', 'brisque_improvement', 'sources_with_error_increase')}, indent=2))


if __name__ == '__main__':
    main()

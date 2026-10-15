"""Evaluate genuine causal outputs; refuse to invent labels or missing runs.

Quality and audio-sensitivity diagnostics can run without labels. TalkNet error
and safeguard measurements require independently reviewed, complete face tracks.
Only descriptive source/track results are produced, never frame-level p-values.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from prepare_causal import ROOT, sha
from run_causal_omnivr import jobs, verify_bundle


def quality(reference, image):
    mse = float(np.mean((reference.astype(np.float64) - image) ** 2))
    return dict(psnr_db=10 * math.log10(255 ** 2 / mse) if mse else None,
                ssim=float(np.mean([structural_similarity(a, b, channel_axis=2, data_range=255) for a, b in zip(reference, image)])))


def mae(a, b):
    return float(np.abs(a.astype(np.float64) - b).mean())


def verified_tracks(annotation, count, width, height):
    if annotation.get('status') != 'VERIFIED' or not str(annotation.get('reviewer') or '').strip():
        raise ValueError('Independent annotations are not verified')
    frames = annotation['frames']
    if len(frames) != count or [f['index'] for f in frames] != list(range(count)):
        raise ValueError('Annotation frame grid mismatch')
    ids = {face['track_id'] for f in frames for face in f['faces']}
    if not ids:
        raise ValueError('No reviewed face tracks')
    tracks = {}
    for identity in sorted(ids):
        boxes, labels = [], []
        for frame in frames:
            faces = [f for f in frame['faces'] if f['track_id'] == identity]
            if len(faces) != 1:
                raise ValueError('Each evaluated track must be present exactly once per frame; reject cuts/occlusion or select another window')
            face = faces[0]
            x1, y1, x2, y2 = face['box_xyxy']
            if any(type(v) is not int for v in (x1, y1, x2, y2)):
                raise ValueError('Face boxes must use integer pixel coordinates')
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError('Invalid face box')
            if face['speaking'] not in (0, 1, None):
                raise ValueError('Invalid speaking label')
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            labels.append(-1 if face['speaking'] is None else face['speaking'])
        if all(y == -1 for y in labels):
            raise ValueError('No known speaking states in track')
        tracks[identity] = (boxes, np.asarray(labels))
    return tracks


def speaker_metrics(labels, probability):
    keep = labels >= 0
    labels, probability = labels[keep], probability[keep]
    if not len(labels):
        return dict(n=0, error_rate=None, false_positive_rate=None, mean_wrong_class_probability=None)
    negative = labels == 0
    return dict(n=len(labels), error_rate=float(np.mean((probability >= .5) != labels)),
        false_positive_rate=float(np.mean(probability[negative] >= .5)) if negative.any() else None,
        mean_wrong_class_probability=float(np.mean(np.where(labels == 0, probability, 1 - probability))))


def safeguards(labels, matching, swapped):
    keep = labels >= 0
    y, p, s = labels[keep], matching[keep], swapped[keep]
    answer = []
    if not len(y):
        return answer
    for coverage in (.5, .75, 1.0):
        n = max(1, int(len(y) * coverage))
        for rule, order in [('conditioning_sensitivity', np.argsort(np.abs(p - s), kind='stable')),
                            ('model_confidence', np.argsort(-np.abs(p - .5), kind='stable'))]:
            indices = order[:n]
            answer.append(dict(rule=rule, coverage=n / len(y), **speaker_metrics(y[indices], p[indices])))
    answer.append(dict(rule='abstain_all', coverage=0, n=0, error_rate=None, false_positive_rate=None, mean_wrong_class_probability=None))
    return answer


class TalkNetCPU:
    def __init__(self):
        import torch
        self.torch = torch
        torch.set_num_threads(2)
        sys.path.insert(0, str(ROOT / 'tools' / 'python'))
        sys.path.insert(0, str(ROOT / 'external' / 'TalkNet-ASD'))
        import python_speech_features
        from model.talkNetModel import talkNetModel
        self.mfcc = python_speech_features.mfcc
        path = ROOT / 'external' / 'TalkNet-ASD' / 'pretrain_TalkSet.model'
        state = torch.load(path, map_location='cpu', weights_only=True)
        self.model = talkNetModel().cpu().eval()
        self.model.load_state_dict({k.removeprefix('model.'): v for k, v in state.items() if k.startswith('model.')}, strict=True)
        self.heads = {}
        for name, size in [('V', 128), ('A', 128), ('AV', 256)]:
            head = torch.nn.Linear(size, 2).eval()
            prefix = 'loss' + name + '.FC.'
            head.load_state_dict({k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}, strict=True)
            self.heads[name] = head
        self.checkpoint_sha256 = sha(path)

    def predict(self, video, source_audio, boxes, fps):
        torch = self.torch
        pcm = np.clip(source_audio * 32768, -32768, 32767).astype(np.int16)
        features = self.mfcc(pcm, 16000, numcep=13, winlen=.025, winstep=.010)
        n = min(int(len(video) / fps * 25), len(features) // 4)
        indices = np.abs(np.arange(len(video))[:, None] / fps - np.arange(n)[None, :] / 25).argmin(axis=0)
        crops = []
        for index in indices:
            x1, y1, x2, y2 = boxes[index]
            gray = cv2.cvtColor(video[index, y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
            crops.append(cv2.resize(gray, (112, 112)))
        ta = torch.from_numpy(features[:4 * n].copy()).float().unsqueeze(0)
        tv = torch.from_numpy(np.stack(crops)).float().unsqueeze(0)
        with torch.inference_mode():
            ae = self.model.forward_audio_frontend(ta)
            ve = self.model.forward_visual_frontend(tv)
            a = self.heads['A'](self.model.forward_audio_backend(ae))
            v = self.heads['V'](self.model.forward_visual_backend(ve))
            ca, cv = self.model.forward_cross_attention(ae, ve)
            av = self.heads['AV'](self.model.forward_audio_visual_backend(ca, cv))
            return {k: torch.softmax(value, dim=-1)[:, 1].numpy() for k, value in [('A', a), ('V', v), ('AV', av)]}, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'causal')
    args = parser.parse_args()
    directory = args.data_dir
    manifest = json.loads((directory / 'manifest.json').read_text())
    verify_bundle(directory, manifest)
    p = manifest['protocol']
    assets = json.loads((ROOT / 'causal_assets.json').read_text())
    weight_hashes = {role + '_sha256': next(r['sha256'] for r in assets['files'] if r['role'] == role)
                     for role in ('base', 'adapter')}
    missing = []
    for row, seed, condition in jobs(manifest):
        stem = f"{row['id']}__seed{seed}__{condition}"
        target, receipt = directory / 'outputs' / (stem + '.npz'), directory / 'outputs' / (stem + '.json')
        if not target.exists() or not receipt.exists():
            missing.append(stem)
            continue
        record = json.loads(receipt.read_text())
        identity = dict(input_id=row['id'], seed=seed, condition=condition,
                        code_revision=p['omnivr']['code_revision'], status='RESTORATION_EXECUTED',
                        generated_audio_used_for_evaluation=False, **weight_hashes)
        if any(record.get(k) != v for k, v in identity.items()):
            raise ValueError(f'Output execution identity mismatch: {stem}')
        expected_audio = None if condition == 'audio_disabled' else row['swapped_audio_sha256' if condition == 'swapped' else 'matching_audio_sha256']
        if (record['output_sha256'] != sha(target) or record['bundle_sha256'] != row['bundle_sha256']
            or record['protocol_sha256'] != manifest['protocol_sha256']
            or record['conditioning_audio_sha256'] != expected_audio
            or record['downstream_audio_sha256'] != row['original_audio_sha256']):
            raise ValueError(f'Output provenance mismatch: {stem}')
    report = dict(status='AWAITING_RESTORATIONS' if missing else 'OUTPUTS_VERIFIED',
        missing_runs=len(missing), expected_runs=manifest['paired_jobs'],
        RQ1='NOT_EVALUATED', RQ2='NOT_EVALUATED', RQ3='NOT_EVALUATED', cases=[],
        score_interpretation='Uncalibrated TalkNet softmax, not calibrated probability of correctness. PSNR/SSIM do not establish perceptual preference or synchronization.',
        inference_unit='Descriptive source/track cases; frames and seeds are not independent subjects.')
    model = None
    if not missing:
        for row in manifest['records']:
            with np.load(directory / row['path']) as data:
                reference, degraded, original_audio = data['reference'], data['degraded'], data['original_audio']
            videos = {'reference': reference, 'degraded': degraded}
            for seed in p['seeds']:
                for condition in p['conditions']:
                    with np.load(directory / 'outputs' / f"{row['id']}__seed{seed}__{condition}.npz") as data:
                        videos[f'{condition}_{seed}'] = data['restored']
            with np.load(directory / 'outputs' / f"{row['id']}__seed0__matching_repeat.npz") as data:
                repeat = data['restored']
            if any(v.shape != reference.shape for v in videos.values()) or repeat.shape != reference.shape:
                raise ValueError('Output frame geometry mismatch')
            case = dict(input_id=row['id'], quality={k: quality(reference, v) for k, v in videos.items() if k != 'reference'},
                same_input_repeat_mae=mae(videos['matching_0'], repeat),
                between_seed_mae=mae(videos['matching_0'], videos['matching_1']),
                audio_intervention_mae={str(seed): mae(videos[f'matching_{seed}'], videos[f'swapped_{seed}']) for seed in p['seeds']}, tracks={})
            annotation = json.loads((directory / (row['sample_id'] + '__annotations.json')).read_text())
            try:
                if annotation.get('source_sha256') != row['source_sha256'] or annotation.get('protocol_sha256') != manifest['protocol_sha256']:
                    raise ValueError('Annotations are not bound to this source and frame protocol')
                tracks = verified_tracks(annotation, p['frames'], p['width'], p['height'])
            except ValueError as error:
                case['speaker_evaluation'] = 'NOT_SCORED: ' + str(error)
                report['cases'].append(case)
                continue
            if model is None:
                model = TalkNetCPU()
            for identity, (boxes, labels) in tracks.items():
                predictions = {}
                for condition, video in videos.items():
                    predictions[condition], indices = model.predict(video, original_audio, boxes, p['fps'])
                if any(not np.array_equal(v['A'], predictions['reference']['A']) for v in predictions.values()):
                    raise ValueError('Downstream audio-only scores changed across conditions')
                y = labels[indices]
                metrics = {c: {head: speaker_metrics(y, probabilities[head]) for head in ('V', 'AV')} for c, probabilities in predictions.items()}
                mitigation = {str(seed): {head: safeguards(y, predictions[f'matching_{seed}'][head], predictions[f'swapped_{seed}'][head]) for head in ('V', 'AV')} for seed in p['seeds']}
                case['tracks'][identity] = dict(metrics=metrics, fixed_coverage_safeguards=mitigation, checkpoint_sha256=model.checkpoint_sha256,
                    annotation_sha256=sha(directory / (row['sample_id'] + '__annotations.json')))
            report['cases'].append(case)
        report['RQ2'] = 'AUDIO_SENSITIVITY_MEASURED; unsupported factual details and exact non-identifiability require separate evidence'
        if any(case['tracks'] for case in report['cases']):
            report['RQ1'] = 'DESCRIPTIVE_ERROR_AND_SCORE_MEASUREMENTS_AVAILABLE; joint quality/error criterion and calibration require interpretation'
            report['RQ3'] = 'FIXED_COVERAGE_BASELINES_MEASURED; not a validated general safeguard'
    (ROOT / 'results' / 'causal_evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()

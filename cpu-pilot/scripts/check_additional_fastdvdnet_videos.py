"""Test FastDVDnet on the three preselected transfer clips, without retuning.

These sources previously had an NLM probe. They are additional to the two-source
FastDVDnet pilot, not a new dataset or a preregistered independent confirmation.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True, type=Path)
    args = parser.parse_args()
    root = args.runtime_root.resolve()
    package = Path(__file__).resolve().parents[1]
    dest = package / 'results/additional_fastdvdnet_videos.json'
    out = package / 'data/additional_fastdvdnet_videos'
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'data/quick_temporal/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    old_path = root / 'results/quick_temporal_pilot.json'
    old = json.loads(old_path.read_text())
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'RUNNING', 'script_sha256': sha(__file__),
        'manifest_sha256': sha(manifest_path), 'prior_result_sha256': sha(old_path),
        'protocol': {
            'selection': 'All three existing quick_temporal manifest entries, unmodified; all silent',
            'source_status': 'Additional to the 2-source FastDVDnet pilot; previously evaluated with NLM',
            'noise_sigma': 25, 'restorer_noise_sigma': 25,
            'input': 'Exact previously saved degraded uint8 arrays; no new corruption or search',
            'restorer': 'Official FastDVDnet clipped-noise weights, CPU',
            'detector': 'Official TalkNet-TalkSet, fixed original audio, threshold 0.5',
            'primary': 'Paired correct-to-incorrect and incorrect-to-correct frame counts',
            'quality': 'Full-reference RGB SSIM only; no perceptual or assessor-consensus claim',
            'negative_result': 'Retain and report every no-change case; no severity tuning',
            'limits': '3 one-second silent tracks; no speaking tracks, Qwen, audio-conditioned restoration, or agent action',
        }, 'manifest': manifest, 'cases': [],
    }
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print('Frozen additional-video protocol saved; loading CPU models', flush=True)
    for directory in ['scripts', 'tools/quality_cpu', 'tools/python', 'external/fastdvdnet']:
        sys.path.insert(0, str(root / directory))
    import av
    import cv2
    import numpy as np
    import torch
    from skimage.metrics import structural_similarity
    from evaluate_causal import TalkNetCPU
    from models import FastDVDnet
    from fastdvdnet import denoise_seq_fastdvdnet
    from reproduce_restoration_comparison import save_video
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cv2.setNumThreads(2)
    model = FastDVDnet().cpu().eval()
    weight = root / 'external/fastdvdnet/model_clipped_noise.pth'
    state = torch.load(weight, map_location='cpu', weights_only=True)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    detector = TalkNetCPU()
    report['weights'] = {'fastdvdnet': sha(weight), 'talknet': detector.checkpoint_sha256}
    report['versions'] = {'python': sys.version, 'torch': torch.__version__, 'cv2': cv2.__version__}
    for entry in manifest['records']:
        sid = entry['source']
        source = root / 'data/quick_temporal' / (sid + '.npz')
        outputs = root / 'data/quick_temporal' / (sid + '_outputs.npz')
        previous = next(c for c in old['cases'] if c['source'] == sid)
        assert sha(source) == entry['sha256']
        assert sha(outputs) == previous['output_sha256']
        with np.load(source) as bundle, np.load(outputs) as cached:
            ref, audio, labels = [bundle[k] for k in ['reference', 'audio', 'labels']]
            captured = cached['degraded']
            assert np.array_equal(ref, cached['reference'])
        x = torch.from_numpy(captured.transpose(0, 3, 1, 2).copy()).float() / 255
        began = time.monotonic()
        with torch.inference_mode():
            y = denoise_seq_fastdvdnet(x, torch.tensor([25 / 255]), 5, model)
        restored = (y.permute(0, 2, 3, 1).numpy().clip(0, 1) * 255).round().astype(np.uint8)
        variants = {'captured': captured, 'fastdvdnet': restored, 'reference': ref}
        case = {'id': sid, 'label': entry['label'], 'source_sha256': sha(source),
                'previous_outputs_sha256': sha(outputs), 'restoration_seconds': time.monotonic() - began,
                'audio_array_sha256': hashlib.sha256(audio.tobytes()).hexdigest(), 'candidates': {}}
        scores = {}
        for name, video in variants.items():
            probs, indices = detector.predict(video, audio, [[0, 0, 128, 128]] * 25, 25)
            assert len(indices) == 24
            if name in ['captured', 'reference']:
                prior = previous['candidates']['degraded' if name == 'captured' else name]['probability']
                np.testing.assert_allclose(probs['AV'], prior, atol=1e-6, rtol=0)
            scores[name] = probs
            case['candidates'][name] = {
                'wrong_frames': int(np.count_nonzero((probs['AV'] >= .5) != labels[indices])),
                'scored_frames': len(indices),
                'ssim': float(np.mean([structural_similarity(a, b, channel_axis=2, data_range=255) for a, b in zip(ref, video)])),
                'probability': probs['AV'].tolist(), 'audio_probability': probs['A'].tolist(),
            }
        assert all(np.array_equal(scores['captured']['A'], scores[k]['A']) for k in variants)
        cc = (scores['captured']['AV'] >= .5) == labels[indices]
        rc = (scores['fastdvdnet']['AV'] >= .5) == labels[indices]
        case.update(new_errors=int(np.count_nonzero(cc & ~rc)),
                    corrections=int(np.count_nonzero(~cc & rc)), audio_only_invariant=True)
        path = out / (sid + '.npz')
        np.savez_compressed(path, **variants, audio=audio, labels=labels)
        case['output_sha256'] = sha(path)
        video = out / (sid + '.mp4')
        save_video(video, variants, scores, indices, entry['label'], audio, sid, cv2, np, av)
        case['preview'] = str(video)
        case['preview_sha256'] = sha(video)
        report['cases'].append(case)
        dest.write_text(json.dumps(report, indent=2) + '\n')
        print(sid, 'errors', case['candidates']['captured']['wrong_frames'], '->',
              case['candidates']['fastdvdnet']['wrong_frames'], flush=True)
    assert len(report['cases']) == 3
    report['status'] = 'COMPLETE'
    report['summary'] = {
        'sources': 3, 'scored_frames': 72,
        'captured_wrong': sum(c['candidates']['captured']['wrong_frames'] for c in report['cases']),
        'restored_wrong': sum(c['candidates']['fastdvdnet']['wrong_frames'] for c in report['cases']),
        'new_errors': sum(c['new_errors'] for c in report['cases']),
        'corrections': sum(c['corrections'] for c in report['cases']),
    }
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()

"""Re-run the eight fixed pilot cases and export inspectable before/after evidence.

This is a reproducibility run on existing sources, not independent validation.
Requires the original runtime checkout with downloaded inputs and model weights.
No API, network, or GPU calls are made.
"""
import argparse
import csv
import hashlib
import html
import json
import platform
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_video(path, variants, scores, indices, label, audio, case_id, cv2, np, av):
    """Preview encoding is separate from the arrays used for evaluation."""
    with av.open(str(path), 'w') as container:
        vs = container.add_stream('libx264', rate=25)
        vs.width, vs.height, vs.pix_fmt = 960, 432, 'yuv420p'
        vs.options = {'crf': '18', 'preset': 'veryfast'}
        aus = container.add_stream('aac', rate=16000)
        aus.layout = 'mono'
        frame_to_score = {int(i): t for t, i in enumerate(indices)}
        for f in range(25):
            canvas = np.full((432, 960, 3), (245, 247, 250), dtype=np.uint8)
            cv2.putText(canvas, case_id, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, .54, (35, 35, 35), 1, cv2.LINE_AA)
            cv2.putText(canvas, 'Dataset label: ' + ('SPEAKING' if label else 'SILENT'), (16, 53), cv2.FONT_HERSHEY_SIMPLEX, .60, (35, 35, 35), 1, cv2.LINE_AA)
            for col, (key, title) in enumerate([('captured', 'Captured noisy'), ('fastdvdnet', 'FastDVDnet restored'), ('reference', 'Source: evaluation only')]):
                left = col * 320 + 16
                cv2.putText(canvas, title, (left, 88), cv2.FONT_HERSHEY_SIMPLEX, .57, (35, 35, 35), 1, cv2.LINE_AA)
                tile = cv2.resize(variants[key][f], (288, 288), interpolation=cv2.INTER_NEAREST)
                canvas[100:388, left:left + 288] = tile
                if f in frame_to_score:
                    p = float(scores[key]['AV'][frame_to_score[f]])
                    prediction = int(p >= .5)
                    color = (175, 35, 35) if prediction != label else (25, 110, 65)
                    caption = f"TalkNet: {'speaking' if prediction else 'silent'} ({p:.3f})"
                else:
                    color, caption = (75, 75, 75), 'Last frame: no aligned ASD score'
                cv2.putText(canvas, caption, (left, 416), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
            frame = av.VideoFrame.from_ndarray(canvas, format='rgb24')
            frame.pts, frame.time_base = f, Fraction(1, 25)
            for packet in vs.encode(frame):
                container.mux(packet)
        for packet in vs.encode():
            container.mux(packet)
        af = av.AudioFrame.from_ndarray(audio.astype(np.float32)[None, :], format='fltp', layout='mono')
        af.sample_rate, af.pts, af.time_base = 16000, 0, Fraction(1, 16000)
        for packet in aus.encode(af):
            container.mux(packet)
        for packet in aus.encode():
            container.mux(packet)
    with av.open(str(path)) as check:
        assert sum(1 for _ in check.decode(video=0)) == 25, 'Preview frame count mismatch'
    with av.open(str(path)) as check:
        assert len(check.streams.audio) == 1
        assert sum(frame.samples for frame in check.decode(audio=0)) >= 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime-root', type=Path, required=True)
    args = ap.parse_args()
    runtime = args.runtime_root.resolve()
    package = Path(__file__).resolve().parents[1]
    output = package / 'data' / 'restoration_comparison'
    output.mkdir(parents=True, exist_ok=True)
    for folder in ['scripts', 'tools/quality_cpu', 'tools/python', 'external/fastdvdnet']:
        sys.path.insert(0, str(runtime / folder))
    print('Loading CPU model runtime', flush=True)
    import av
    import cv2
    import numpy as np
    import torch
    from skimage.metrics import structural_similarity
    from evaluate_causal import TalkNetCPU
    from models import FastDVDnet
    from fastdvdnet import denoise_seq_fastdvdnet
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cv2.setNumThreads(2)

    saved_path = runtime / 'results/agent_pilot_measurements.json'
    saved = json.loads(saved_path.read_text())
    records = {r['id']: r for r in saved['manifest']['records']}
    protocol_path = runtime / 'agent_pilot_protocol.json'
    assert sha(protocol_path) == saved['protocol_sha256']
    assert sha(runtime / 'scripts/agent_pilot_measure.py') == saved['script_sha256']
    weight = runtime / 'external/fastdvdnet/model_clipped_noise.pth'
    assert sha(weight) == saved['weights']['fastdvdnet']
    model = FastDVDnet().cpu().eval()
    state = torch.load(weight, map_location='cpu', weights_only=True)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    del state
    detector = TalkNetCPU()
    assert detector.checkpoint_sha256 == saved['weights']['talknet']
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  status='RUNNING', run_type='same-source reproducibility and media export',
                  model='Official FastDVDnet, clipped-noise checkpoint, true noise sigma',
                  detector='Official TalkNet, CPU, decision threshold 0.5',
                  audio_conditioned_restoration=False, omnivr_executed=False,
                  new_independent_sources=0, script_sha256=sha(__file__),
                  saved_measurements_sha256=sha(saved_path), protocol_sha256=sha(protocol_path),
                  weights={k: saved['weights'][k] for k in ['fastdvdnet', 'talknet']},
                  versions=dict(python=platform.python_version(), torch=torch.__version__, cv2=cv2.__version__, av=av.__version__),
                  quality_note='SSIM recomputed. MUSIQ/BRISQUE copied from prior run only after byte-identical arrays are verified.',
                  preview_note='Nearest-neighbor display enlargement and H.264/AAC preview encoding are not evaluated inputs.',
                  cases=[])
    dest = package / 'results/restoration_before_after_reproduction.json'
    dest.parent.mkdir(exist_ok=True)
    started = time.monotonic()
    for old in saved['cases']:
        source_row = records[old['id'].rsplit('__sigma', 1)[0]]
        source_path = Path(source_row['path'])
        assert sha(source_path) == source_row['sha256']
        assert sha(old['outputs']) == old['output_sha256']
        with np.load(source_path) as bundle:
            reference, audio, labels = (bundle[k].copy() for k in ['reference', 'audio', 'labels'])
        rng = np.random.default_rng(old['seed'])
        captured = np.clip(np.rint(reference.astype(float) + rng.normal(0, old['sigma'], reference.shape)), 0, 255).astype(np.uint8)
        t0 = time.monotonic()
        x = torch.from_numpy(captured.transpose(0, 3, 1, 2).copy()).float() / 255
        with torch.inference_mode():
            y = denoise_seq_fastdvdnet(x, torch.tensor([old['sigma'] / 255]), 5, model)
        restored = (y.permute(0, 2, 3, 1).numpy().clip(0, 1) * 255).round().astype(np.uint8)
        restoration_seconds = time.monotonic() - t0
        variants = dict(captured=captured, fastdvdnet=restored, reference=reference)
        with np.load(old['outputs']) as previous:
            for name, video in variants.items():
                assert np.array_equal(video, previous[name]), f'{old["id"]}: {name} changed since previous run'
        scores, metrics, indices = {}, {}, None
        for name, video in variants.items():
            prob, aligned = detector.predict(video, audio, [[0, 0, 128, 128]] * 25, 25)
            assert len(aligned) == 24
            if indices is not None:
                assert np.array_equal(aligned, indices)
            indices = aligned
            previous = old['candidates'][name]
            assert np.allclose(prob['AV'], previous['probability'], atol=1e-6, rtol=0), 'ASD scores not reproduced'
            truth = labels[indices]
            correct = (prob['AV'] >= .5) == truth
            scores[name] = prob
            metrics[name] = dict(wrong_frames=int((~correct).sum()), scored_frames=len(indices),
                                 error_rate=float((~correct).mean()),
                                 probability=prob['AV'].tolist(),
                                 ssim=float(np.mean([structural_similarity(a, b, channel_axis=2, data_range=255) for a, b in zip(reference, video)])),
                                 musiq_previous_verified=previous['musiq'], brisque_previous_verified=previous['brisque'])
        assert np.array_equal(scores['captured']['A'], scores['fastdvdnet']['A'])
        captured_correct = (scores['captured']['AV'] >= .5) == labels[indices]
        restored_correct = (scores['fastdvdnet']['AV'] >= .5) == labels[indices]
        case = dict(id=old['id'], source=old['source'], label=old['label'], sigma=old['sigma'], seed=old['seed'],
                    source_bundle_sha256=sha(source_path), prior_output_sha256=sha(old['outputs']),
                    audio_array_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
                    output_arrays_identical=True, predictions_match_atol=1e-6,
                    audio_only_predictions_identical=True, restoration_seconds=restoration_seconds,
                    indices=indices.tolist(), candidates=metrics,
                    new_errors=int((captured_correct & ~restored_correct).sum()),
                    corrections=int((~captured_correct & restored_correct).sum()))
        media = output / (old['id'] + '.mp4')
        save_video(media, variants, scores, indices, old['label'], audio, old['id'], cv2, np, av)
        case['video'], case['video_sha256'] = str(media), sha(media)
        report['cases'].append(case)
        dest.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(old['id'], 'errors', metrics['captured']['wrong_frames'], '->', metrics['fastdvdnet']['wrong_frames'],
              'new', case['new_errors'], 'corrected', case['corrections'], 'verified/exported', flush=True)
    assert len(report['cases']) == 8
    report['status'] = 'PASS_REPRODUCED'
    report['wall_seconds'] = time.monotonic() - started
    report['summary'] = dict(cases=8, independent_sources=len({c['source'] for c in report['cases']}),
                            scored_frames=sum(c['candidates']['captured']['scored_frames'] for c in report['cases']),
                            captured_wrong=sum(c['candidates']['captured']['wrong_frames'] for c in report['cases']),
                            restored_wrong=sum(c['candidates']['fastdvdnet']['wrong_frames'] for c in report['cases']),
                            new_errors=sum(c['new_errors'] for c in report['cases']),
                            corrections=sum(c['corrections'] for c in report['cases']))
    dest.write_text(json.dumps(report, indent=2), encoding='utf-8')
    flat = []
    for case in report['cases']:
        row = {k: case[k] for k in ['id', 'source', 'label', 'sigma', 'new_errors', 'corrections']}
        for name in ['captured', 'fastdvdnet', 'reference']:
            for metric in ['wrong_frames', 'ssim', 'musiq_previous_verified', 'brisque_previous_verified']:
                row[name + '_' + metric] = case['candidates'][name][metric]
        flat.append(row)
    with (dest.parent / 'restoration_before_after_reproduction.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    make_review(report, output, dest.parent)
    print(json.dumps(report['summary']), flush=True)
    print('Review page:', output / 'index.html', flush=True)


def make_review(report, output, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True, sharey=True)
    cards = []
    for case, ax in zip(report['cases'], axes.flat):
        for name, color, title in [('captured', '#356bc4', 'Captured'), ('fastdvdnet', '#ce7530', 'FastDVDnet'), ('reference', '#708071', 'Source reference')]:
            ax.plot(np.array(case['indices']) / 25, case['candidates'][name]['probability'], color=color, label=title, lw=1.6)
        ax.axhline(.5, color='#555', ls='--', lw=.8)
        ax.set_ylim(-.03, 1.03)
        ax.grid(alpha=.16)
        a, b = case['candidates']['captured']['wrong_frames'], case['candidates']['fastdvdnet']['wrong_frames']
        ax.set_title(f"{case['source']} | {'speaking' if case['label'] else 'silent'} | noise {case['sigma']}\nWrong frames: {a}/24 to {b}/24", fontsize=10)
        cards.append(f'<article><h2>{html.escape(case["id"])}</h2><p>Dataset label: {"speaking" if case["label"] else "silent"}. Errors: {a}/24 → {b}/24. New errors: {case["new_errors"]}; corrections: {case["corrections"]}.</p><video controls loop preload="metadata" src="{html.escape(Path(case["video"]).name)}"></video></article>')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, .945))
    fig.suptitle('Actual restoration and downstream inference: all eight existing pilot cases', fontsize=15, y=.985)
    fig.supxlabel('Time within clip (seconds)', y=.04)
    fig.supylabel('Uncalibrated TalkNet speaking score')
    fig.text(.5, .012, 'Same-source reproduction, not new validation. Original audio fixed. Dashed line: decision threshold 0.5.', ha='center', fontsize=9)
    fig.tight_layout(rect=(.02, .075, 1, .92))
    for suffix in ['png', 'pdf']:
        fig.savefig(results / ('restoration_before_after_traces.' + suffix), dpi=160)
    plt.close(fig)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Restoration: before and after</title><style>body{font:16px/1.5 system-ui;background:#f4f6fa;color:#172033;max-width:1100px;margin:40px auto;padding:0 20px}h1{font-size:30px}article{background:white;padding:22px;margin:24px 0;border-radius:12px}h2{font-size:19px}video{width:100%;background:#111}p{max-width:950px}.note{border-left:4px solid #356bc4;padding:12px 20px;background:white}</style><h1>Restoration and speaker predictions: before and after</h1><p class="note">All eight cases from the existing labelled pilot were restored and scored again. Arrays reproduced exactly and speaker scores matched within 0.000001. This is a reproduction, not eight new independent videos. FastDVDnet receives video only. TalkNet receives the same original audio for every variant. OmniVR was not run.</p><p>Each clip shows captured noisy video, actual FastDVDnet restoration, and the source reference used only for evaluation. Predictions refer to the visible face. A score above 0.5 is a speaking prediction, not calibrated confidence. Visual mouth artifacts or human-perceived improvements are not established by these scores.</p>'''
    page += ''.join(cards)
    page += '<p>Media previews use display enlargement and H.264/AAC encoding. Evaluations use the original arrays and unchanged waveform. The 25th frame has no aligned TalkNet score; it is not counted.</p></html>'
    (output / 'index.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()

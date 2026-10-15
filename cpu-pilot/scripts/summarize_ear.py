"""Summarize executed still-image interventions, without speaker claims."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]


def main():
    probe = json.loads((ROOT / 'results' / 'ear_cpu_probe.json').read_text())
    directory = ROOT / 'data' / 'ear_cpu'
    cases = []
    canvas = Image.new('RGB', (5 * 160, len(probe['examples']) * 190 + 28), 'white')
    draw = ImageDraw.Draw(canvas)
    titles = ['Reference', '8x8 input', 'Visual-only SR', 'Matching audio', 'Swapped audio']
    for col, label in enumerate(titles):
        draw.text((col * 160 + 4, 7), label, fill='black')
    for row, record in enumerate(probe['examples']):
        with np.load(directory / (record['example'] + '.npz')) as data:
            images = [data['reference'], np.asarray(Image.fromarray(data['degraded']).resize((128, 128), Image.Resampling.NEAREST)),
                      data['video_only'], data['matching_audio'], data['swapped_audio']]
            if not np.array_equal(data['matching_audio'], data['matching_repeat']):
                raise ValueError('Identical-input repeat differs; investigate before causal interpretation')
            measures = {}
            for label, image in zip(titles[1:], images[1:]):
                mse = float(np.mean((images[0].astype(float) - image) ** 2))
                measures[label] = dict(psnr_db=float(10 * np.log10(255 ** 2 / mse)) if mse else None,
                                       ssim=float(structural_similarity(images[0], image, channel_axis=2, data_range=255)))
            cases.append(dict(example=record['example'], quality=measures,
                audio_swap_rgb_mae=record['audio_swap_rgb_mae'], identical_repeat=True))
            for col, image in enumerate(images):
                canvas.paste(Image.fromarray(image).resize((160, 160)), (160 * col, 28 + row * 190))
                draw.text((160 * col + 4, 191 + row * 190), record['example'], fill='black')
    canvas.save(directory / 'comparison.png')
    report = dict(status='MEASURED_CPU_STILL_IMAGE_INTERVENTION', cases=cases,
        conclusion='Audio changes restored pixels with visual input held fixed; identical repeats match on these cases.',
        original_RQ1='Not tested: no temporal speaker-error/confidence experiment.',
        original_RQ2='Limited mechanism evidence only: audio sensitivity, not proof of visually unsupported facts or non-identifiability.',
        original_RQ3='Not tested: no labeled safeguard evaluation on these still examples.')
    (ROOT / 'results' / 'ear_cpu_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""Check published-label/audio alignment and render a local input contact sheet."""
import csv
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.io import wavfile
from prepare_causal import ROOT


def main():
    directory = ROOT / 'data' / 'unitalk'
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows = []
    sheet = Image.new('RGB', (640, len(manifest['records']) * 152), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(manifest['records']):
        track = [r for r in csv.reader((directory / (item['source'] + '.csv')).read_text().splitlines()) if r[7] == item['entity']]
        track.sort(key=lambda r: float(r[1]))
        with zipfile.ZipFile(directory / (item['source'] + '_audio.zip')) as z:
            name = next(n for n in z.namelist() if Path(n).stem == item['entity'] and n.endswith('.wav'))
            rate, audio = wavfile.read(io.BytesIO(z.read(name)))
        expected = float(track[-1][1]) - float(track[0][1]) + .04
        error = len(audio) / rate - expected
        if abs(error) > .05:
            raise ValueError(f'Audio/track duration discrepancy: {item["id"]}: {error}')
        target = [r for r in track if item['target_start'] - .0001 <= float(r[1]) < item['target_start'] + .9999]
        if len(target) != 25 or any(int(r[8]) != item['label'] or r[6] != ('SPEAKING_AUDIBLE' if item['label'] else 'NOT_SPEAKING') for r in target):
            raise ValueError('Published label inconsistency')
        with np.load(directory / item['path']) as data:
            reference = data['reference']
            for j, frame in enumerate((0, 12, 24)):
                sheet.paste(Image.fromarray(reference[frame]), (240 + j * 132, index * 152 + 20))
            rms = float(np.sqrt(np.mean(data['original_audio'] ** 2)))
        draw.text((4, index * 152 + 25), item['id'], fill='black')
        draw.text((4, index * 152 + 48), f"Published label: {item['label']}  t={item['target_start']}", fill='black')
        rows.append(dict(id=item['id'], label=item['label'], audio_duration_error_seconds=error,
                         source_audio_rms=rms, published_label_consistent=True))
    sheet.save(directory / 'input_contact_sheet.png')
    (ROOT / 'results' / 'unitalk_input_audit.json').write_text(json.dumps(dict(status='PASSED', checks=rows,
        limitation='Duration/grid and published-label consistency checks, not independent human re-annotation.'), indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()

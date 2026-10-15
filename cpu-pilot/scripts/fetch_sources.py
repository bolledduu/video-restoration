"""Download only the three manifest clips; pin metadata and record byte hashes."""
import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def get(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()

def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    dataset = manifest['dataset']
    revision = manifest.get('dataset_revision')
    if not revision:
        revision = json.loads(get(f'https://huggingface.co/api/datasets/{dataset}'))['sha']
    base = f'https://huggingface.co/datasets/{dataset}/resolve/{revision}/'
    raw = ROOT / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    card = get(base + 'README.md')
    (raw / 'dataset_card.md').write_bytes(card)
    rows = list(csv.DictReader(io.StringIO(get(base + 'test.csv').decode('utf-8'))))
    records = []
    for sample in manifest['samples']:
        row = next(r for r in rows if r['question_id'] == sample['question_id'])
        assert row['video_id'] == sample['video_id']
        assert row['audio_visual_path'] == sample['file']
        assert row['answer'] == sample['published_answer']
        record = {'sample_id': sample['sample_id'], 'dataset_revision': revision,
                  'source_url': base + sample['file'], 'published_metadata': row,
                  'independent_annotation_verified': False}
        try:
            payload = get(record['source_url'])
            (raw / (sample['sample_id'] + '.mp4')).write_bytes(payload)
            record.update(status='downloaded', bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        except Exception as exc:
            record.update(status='download_failed', error=str(exc))
        records.append(record)
        print(sample['sample_id'], record['status'], flush=True)
    (ROOT / 'results' / 'source_downloads.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(card.decode('utf-8'))

if __name__ == '__main__':
    main()

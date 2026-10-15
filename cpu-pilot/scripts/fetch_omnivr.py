"""Download three predetermined public OmniVR triplets, without credentials."""
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = 'xin1u/OmniVR'
REVISION = '3a58cc2cf24262ce7b81e053454c462614fd5eae'


def get(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def main():
    folder = ROOT / 'data/omnivr'
    folder.mkdir(parents=True, exist_ok=True)
    base = f'https://huggingface.co/{REPO}/resolve/{REVISION}/'
    inventory = json.loads(get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'))
    entries = {x['rfilename']: x for x in inventory['siblings']}
    manifest = json.loads(get(base + 'omnibench/manifest.json'))
    samples = sorted((x for x in manifest['clips'] if x['group'] == 'with_gt'), key=lambda x: x['name'])[:3]
    jobs = []
    for sample in samples:
        for condition, remote in [('reference', 'omnibench/' + sample['gt_path']),
                                  ('degraded', 'omnibench/' + sample['lq_path']),
                                  ('omnivr_author_output', 'predictions/withgt_step02400_neg_cfg/' + sample['name'])]:
            jobs.append((sample, condition, remote))

    def download(job):
        sample, condition, remote = job
        local = folder / condition / sample['name']
        local.parent.mkdir(exist_ok=True)
        expected = entries[remote]['lfs']['sha256']
        if not local.exists() or hashlib.sha256(local.read_bytes()).hexdigest() != expected:
            local.write_bytes(get(base + remote))
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        assert digest == expected
        return dict(sample=sample, condition=condition, url=base + remote,
                    local_path=local.relative_to(ROOT).as_posix(), bytes=local.stat().st_size,
                    sha256=digest, hash_verified=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, jobs))
    result = dict(repository=REPO, revision=REVISION,
                  selection='First three with_gt records in filename order, chosen before inspecting predictions.',
                  restoration_provenance='Published author outputs; no local OmniVR inference.',
                  authentication_used=False, source_media_redistribution='Not committed; source rights not independently verified.',
                  records=records)
    (ROOT / 'results/omnivr_sources.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(files=len(records), bytes=sum(x['bytes'] for x in records), hashes_verified=True)))


if __name__ == '__main__':
    main()

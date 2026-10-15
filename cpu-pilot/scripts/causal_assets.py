"""Record exact GPU asset hashes; download only with explicit --download.

Gemma requires the user's independently approved Hugging Face access. This script
does not accept licenses, use a pasted token, rent hardware, or start inference.
"""
import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = [
    ('base', 'Lightricks/LTX-2.3', '3c6a4e66e5d0a684231950b9c74dd4ded7b6fadc', 'ltx-2.3-22b-dev.safetensors'),
    ('adapter', 'xin1u/OmniVR', '3a58cc2cf24262ce7b81e053454c462614fd5eae', 'weights/ominivr_lora_step_02400.safetensors'),
    ('text_encoder', 'google/gemma-3-12b-it-qat-q4_0-unquantized', '68f7ee4fbd59087436ada77ed2d62f373fdd4482', None),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', type=Path, help='Download weights into this directory on the target machine; otherwise metadata only')
    args = parser.parse_args()
    records = []
    for role, repo, revision, filename in SPECS:
        url = f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
        with urllib.request.urlopen(url, timeout=60) as response:
            model = json.load(response)
        for entry in model['siblings']:
            name = entry['rfilename']
            if filename is not None and name != filename:
                continue
            if filename is None and not name.endswith(('.json', '.safetensors', '.model')):
                continue
            records.append(dict(role=role, repository=repo, revision=revision, file=name,
                bytes=entry['size'], sha256=entry.get('lfs', {}).get('sha256'), git_blob_sha1=entry['blobId']))
    if not all(any(r['role'] == role for r in records) for role in ('base', 'adapter', 'text_encoder')):
        raise ValueError('Required asset missing from pinned release')
    manifest = dict(files=records, total_bytes=sum(r['bytes'] for r in records), text_encoder_access='manual HF approval required',
                    code_revision='81d00ff1cb271ef79210a4082e2058b33ddc310d')
    (ROOT / 'causal_assets.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(total_bytes=manifest['total_bytes'], files=len(records), download_requested=args.download is not None, access=manifest['text_encoder_access']), indent=2))
    if args.download is not None:
        from huggingface_hub import hf_hub_download
        for record in records:
            hf_hub_download(repo_id=record['repository'], revision=record['revision'], filename=record['file'],
                            local_dir=args.download / record['role'])


if __name__ == '__main__':
    main()

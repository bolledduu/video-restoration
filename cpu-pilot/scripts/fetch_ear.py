"""Fetch pinned public official Ear-for-Face-SR weights; no authentication."""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'python'))
import gdown

FILES = {
    'model-0.index': ('1AdJbkiu9wTEQ3PHPQIsRDlENv-JZNdi-', '30716786876a9c969844ad31b9cadca8f52fa6eb5e073a84c8fd6384174c1854'),
    'model-0.data-00000-of-00001': ('1UsXOrcMkoP6qlp700MGCK70iQ2dX7ieE', 'fc99bed110e02f0507fa84c6043f759def33d681f22aef4dcdc14a3e1cc5bda5'),
    'network-snapshot-030929.pkl': ('1Gx4XvqjpFahX0eZNgL_uMG9jFNDLc_JH', 'a1d1fa6d36709ba789b823c8458cb746636d8e2d8c1bbed17bda0b2d864b1cd8'),
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    dest = ROOT / 'external' / 'ear_for_face_super_resolution' / 'checkpoint'
    dest.mkdir(parents=True, exist_ok=True)
    for name, (identity, expected) in FILES.items():
        path = dest / name
        if not path.exists():
            gdown.download(id=identity, output=str(path), quiet=False)
        if sha(path) != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
        print(name, 'verified')


if __name__ == '__main__':
    main()

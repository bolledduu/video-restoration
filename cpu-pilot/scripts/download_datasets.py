"""Download datasets for audiovisual restoration and speaker-attribution experiments.

Downloads:
  1. AVA-ActiveSpeaker: 25 YouTube videos (hash-ordered from manifest.json)
     - 5 Dev sources + 20 Test sources (source-disjoint split)
  2. AMI Meeting Corpus: 15 meetings (close-up camera videos + headset audio)
     - 3 Dev meetings + 12 Test meetings (disjoint participant groups)

Usage:
  python download_datasets.py --dataset ava
  python download_datasets.py --dataset ami
  python download_datasets.py --dataset all
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# ============================================================================
# AVA-ActiveSpeaker Setup
# ============================================================================

# 25 videos: first 25 from the hash-ordered source_order in manifest.json
# 5 dev + 20 test (source-disjoint split)
AVA_SOURCES = [
    "4ZpjKfu6Cl8",   # 1  - dev
    "IKdBLciu_-A",    # 2  - dev
    "JNb4nWexD0I",    # 3  - dev
    "a5mEmM6w_ks",    # 4  - dev
    "xJmRNZVDDCY",    # 5  - dev
    "P60OxWahxBQ",    # 6  - test
    "rJKeqfTlAeY",    # 7  - test
    "ax3q-RkVIt4",    # 8  - test
    "8nO5FFbIAog",    # 9  - test
    "HV0H6oc4Kvs",    # 10 - test
    "2DUITARAsWQ",    # 11 - test
    "PmElx9ZVByw",    # 12 - test
    "zC5Fh2tTS1U",    # 13 - test
    "_7oWZq_s_Sk",    # 14 - test
    "qrkff49p4E4",    # 15 - test
    "kMy-6RtoOVU",    # 16 - test
    "HKjR70GCRPE",    # 17 - test
    "KHHgQ_Pe4cI",    # 18 - test
    "IzvOYVMltkI",    # 19 - test
    "1j20qq1JyX4",    # 20 - test
    "UrsCy6qIGoo",    # 21 - test
    "j5jmjhGBW44",    # 22 - test
    "5milLu-6bWI",    # 23 - test
    "z-fsLpGHq6o",    # 24 - test
    "LgBQlW6OTr0",    # 25 - test
]

AVA_DEV_SOURCES = set(AVA_SOURCES[:5])
AVA_TEST_SOURCES = set(AVA_SOURCES[5:])


def extract_ava_annotations():
    """Extract AVA annotation tarball if not already extracted."""
    tar_path = DATA_DIR / "ava" / "ava_activespeaker_val_v1.0.tar.bz2"
    ann_dir = DATA_DIR / "ava" / "annotations"

    if ann_dir.exists() and any(ann_dir.iterdir()):
        print(f"  [OK] AVA annotations already extracted at {ann_dir}", flush=True)
        return True

    if not tar_path.exists():
        print(f"  [ERROR] AVA annotation tarball not found: {tar_path}", flush=True)
        return False

    print(f"  Extracting {tar_path.name}...", flush=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(path=ann_dir)
        print(f"  [OK] Extracted annotations to {ann_dir}", flush=True)
        return True
    except Exception as e:
        print(f"  [ERROR] Extraction failed: {e}", flush=True)
        return False


def download_ava_videos(max_retries=2):
    """Download 25 AVA-ActiveSpeaker videos using yt-dlp."""
    video_dir = DATA_DIR / "ava" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    status = {"downloaded": [], "skipped": [], "failed": []}

    print("=" * 70, flush=True)
    print("AVA-ActiveSpeaker: Downloading 25 source videos", flush=True)
    print(f"  Dev sources (5):  {', '.join(sorted(AVA_DEV_SOURCES))}", flush=True)
    print(f"  Test sources (20): {len(AVA_TEST_SOURCES)} sources", flush=True)
    print(f"  Output directory:  {video_dir}", flush=True)
    print("=" * 70, flush=True)

    # Extract annotations first
    extract_ava_annotations()

    for i, vid_id in enumerate(AVA_SOURCES, 1):
        split = "dev" if vid_id in AVA_DEV_SOURCES else "test"

        # Check if already downloaded
        existing = list(video_dir.glob(f"{vid_id}.*"))
        # Exclude temporary download files like .part, .ytdl
        valid_existing = [f for f in existing if not f.name.endswith((".part", ".ytdl")) and f.stat().st_size > 1024 * 1024]
        if valid_existing:
            size_mb = valid_existing[0].stat().st_size / (1024 * 1024)
            print(f"  [{i:2d}/25] {vid_id} ({split}) - already exists: {valid_existing[0].name} ({size_mb:.1f} MB)", flush=True)
            status["skipped"].append(vid_id)
            continue

        url = f"https://www.youtube.com/watch?v={vid_id}"
        output_template = str(video_dir / f"{vid_id}.%(ext)s")

        print(f"  [{i:2d}/25] {vid_id} ({split}) - downloading...", flush=True)

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                cmd = [
                    sys.executable, "-m", "yt_dlp",
                    "--no-playlist",
                    "--format", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
                    "--merge-output-format", "mp4",
                    "--output", output_template,
                    "--no-overwrites",
                    "--retries", "3",
                    "--socket-timeout", "30",
                    url,
                ]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout per video
                )

                if proc.returncode == 0:
                    downloaded = [f for f in video_dir.glob(f"{vid_id}.*") if not f.name.endswith((".part", ".ytdl"))]
                    if downloaded:
                        size_mb = downloaded[0].stat().st_size / (1024 * 1024)
                        print(f"           [OK] Downloaded: {downloaded[0].name} ({size_mb:.1f} MB)", flush=True)
                        status["downloaded"].append(vid_id)
                        success = True
                        break
                    else:
                        print(f"           [?] Success exit code but output file missing", flush=True)
                else:
                    err_snip = proc.stderr[-200:].strip() if proc.stderr else "unknown error"
                    print(f"           [Attempt {attempt}/{max_retries}] error: {err_snip}", flush=True)

            except subprocess.TimeoutExpired:
                print(f"           [Attempt {attempt}/{max_retries}] timed out (10 min)", flush=True)
            except Exception as e:
                print(f"           [Attempt {attempt}/{max_retries}] exception: {e}", flush=True)

        if not success:
            # Fallback to official Google Research AVA AWS S3 mirror
            print(f"           [Fallback] Trying official AWS S3 mirror for {vid_id}...", flush=True)
            for ext in ["mkv", "mp4"]:
                s3_url = f"https://s3.amazonaws.com/ava-dataset/trainval/{vid_id}.{ext}"
                dest_file = video_dir / f"{vid_id}.{ext}"
                if download_stream(s3_url, dest_file, f"{vid_id}.{ext}"):
                    status["downloaded"].append(vid_id)
                    success = True
                    break

        if not success:
            print(f"           [FAILED] could not download {vid_id}", flush=True)
            status["failed"].append(vid_id)


    print(flush=True)
    print(f"  AVA Summary: {len(status['downloaded'])} downloaded, "
          f"{len(status['skipped'])} existing, "
          f"{len(status['failed'])} failed", flush=True)
    return status


# ============================================================================
# AMI Meeting Corpus Setup
# ============================================================================

# 15 meetings selected from scenario-based subset
# Grouped by scenario series to ensure source-disjoint splits
# ES = Edinburgh Scenario, IS = Idiap Scenario, TS = TNO Scenario
AMI_MEETINGS = {
    # Development (3 meetings from 3 independent groups)
    "dev": [
        "ES2002a",   # Edinburgh scenario group 2, session a
        "IS1000a",   # Idiap scenario group 0, session a
        "TS3003a",   # TNO scenario group 3, session a
    ],
    # Test (12 meetings from disjoint groups - no shared participants)
    "test": [
        "ES2003a",   # Edinburgh group 3
        "ES2004a",   # Edinburgh group 4
        "ES2005a",   # Edinburgh group 5
        "ES2006a",   # Edinburgh group 6
        "ES2007a",   # Edinburgh group 7
        "ES2008a",   # Edinburgh group 8
        "IS1001a",   # Idiap group 1
        "IS1003a",   # Idiap group 3
        "IS1004a",   # Idiap group 4 (verified on mirror)
        "TS3004a",   # TNO group 4
        "TS3005a",   # TNO group 5
        "TS3006a",   # TNO group 6
    ],
}

AMI_MIRROR_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus"
AMI_ANNOTATION_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"


def download_stream(url, dest_path, desc=""):
    """Download a single file via HTTP with streaming and resume check."""
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        if size_mb > 0.05:  # Valid non-empty file
            print(f"      [exists] {dest_path.name} ({size_mb:.1f} MB)", flush=True)
            return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "WACV-Research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response, open(temp_path, "wb") as out_file:
                total_len = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1 MB chunk
                t0 = time.time()
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)

            # Atomic rename on success
            if temp_path.exists():
                temp_path.replace(dest_path)
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"      [OK] {dest_path.name} ({size_mb:.1f} MB)", flush=True)
            return True

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if attempt == 3:
                print(f"      [FAILED] {dest_path.name} - {e}", flush=True)
                return False
            time.sleep(2)
    return False


def download_ami_annotations():
    """Download and extract the complete AMI manual annotation archive."""
    ann_dir = DATA_DIR / "ami" / "annotations"
    if ann_dir.exists() and any(ann_dir.iterdir()):
        print("  [OK] AMI annotations already extracted", flush=True)
        return True

    ann_dir.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "ami" / "ami_public_manual_1.6.2.zip"

    print("  Downloading AMI public manual annotation zip archive...", flush=True)
    if download_stream(AMI_ANNOTATION_URL, zip_path, "AMI annotations zip"):
        print("  Extracting annotations...", flush=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(ann_dir)
            print("  [OK] Annotations extracted successfully", flush=True)
            return True
        except Exception as e:
            print(f"  [ERROR] Unzipping failed: {e}", flush=True)
            return False
    return False


def download_ami_meetings():
    """Download 15 AMI meetings: 4 close-up cameras + 4 headset mics + 1 mix per meeting."""
    ami_dir = DATA_DIR / "ami"
    ami_dir.mkdir(parents=True, exist_ok=True)

    all_meetings = AMI_MEETINGS["dev"] + AMI_MEETINGS["test"]
    status = {"downloaded": [], "partial": [], "failed": []}

    print("=" * 70, flush=True)
    print("AMI Meeting Corpus: Downloading 15 meetings", flush=True)
    print(f"  Dev meetings (3):  {', '.join(AMI_MEETINGS['dev'])}", flush=True)
    print(f"  Test meetings (12): {', '.join(AMI_MEETINGS['test'])}", flush=True)
    print(f"  Output directory:   {ami_dir}", flush=True)
    print("=" * 70, flush=True)

    # Annotations first
    download_ami_annotations()

    for idx, meeting_id in enumerate(all_meetings, 1):
        split = "dev" if meeting_id in AMI_MEETINGS["dev"] else "test"
        meeting_dir = ami_dir / "meetings" / meeting_id
        meeting_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{idx:2d}/15] {meeting_id} ({split})", flush=True)

        # Plan the 9 files for this meeting: 4 closeups + 4 headsets + 1 mix
        target_files = []
        for c in range(1, 5):
            fn = f"{meeting_id}.Closeup{c}.avi"
            target_files.append((f"{AMI_MIRROR_BASE}/{meeting_id}/video/{fn}", meeting_dir / fn, f"video {c}"))
        for h in range(4):
            fn = f"{meeting_id}.Headset-{h}.wav"
            target_files.append((f"{AMI_MIRROR_BASE}/{meeting_id}/audio/{fn}", meeting_dir / fn, f"audio {h}"))
        fn = f"{meeting_id}.Mix-Headset.wav"
        target_files.append((f"{AMI_MIRROR_BASE}/{meeting_id}/audio/{fn}", meeting_dir / fn, "mix audio"))

        files_ok = 0
        for url, dest, desc in target_files:
            if download_stream(url, dest, desc):
                files_ok += 1

        total = len(target_files)
        if files_ok == total:
            status["downloaded"].append(meeting_id)
            print(f"    [OK] Meeting complete ({files_ok}/{total} files)", flush=True)
        elif files_ok > 0:
            status["partial"].append(meeting_id)
            print(f"    [PARTIAL] Meeting partially downloaded ({files_ok}/{total} files)", flush=True)
        else:
            status["failed"].append(meeting_id)
            print(f"    [FAILED] Meeting download failed (0/{total} files)", flush=True)

    print(flush=True)
    print(f"  AMI Summary: {len(status['downloaded'])} complete, "
          f"{len(status['partial'])} partial, "
          f"{len(status['failed'])} failed", flush=True)
    return status


# ============================================================================
# Manifest Generation
# ============================================================================

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()


def generate_download_manifest():
    """Generate a manifest recording downloaded assets and their SHA256 hashes."""
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ava": {
            "dev_sources": sorted(AVA_DEV_SOURCES),
            "test_sources": sorted(AVA_TEST_SOURCES),
            "videos": {},
        },
        "ami": {
            "dev_meetings": AMI_MEETINGS["dev"],
            "test_meetings": AMI_MEETINGS["test"],
            "meetings": {},
        },
    }

    # AVA videos
    ava_dir = DATA_DIR / "ava" / "videos"
    if ava_dir.exists():
        for vid_id in AVA_SOURCES:
            matches = [f for f in ava_dir.glob(f"{vid_id}.*") if not f.name.endswith((".part", ".ytdl"))]
            if matches:
                f = matches[0]
                manifest["ava"]["videos"][vid_id] = {
                    "filename": f.name,
                    "size_bytes": f.stat().st_size,
                    "sha256": sha256_file(f),
                }

    # AMI meetings
    ami_meetings_dir = DATA_DIR / "ami" / "meetings"
    if ami_meetings_dir.exists():
        for mid in AMI_MEETINGS["dev"] + AMI_MEETINGS["test"]:
            mdir = ami_meetings_dir / mid
            if mdir.exists():
                manifest["ami"]["meetings"][mid] = {}
                for f in sorted(mdir.iterdir()):
                    if f.is_file() and not f.name.endswith(".part"):
                        manifest["ami"]["meetings"][mid][f.name] = {
                            "size_bytes": f.stat().st_size,
                            "sha256": sha256_file(f),
                        }

    manifest_path = DATA_DIR / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest saved: {manifest_path}", flush=True)
    return manifest


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download datasets for experiments")
    parser.add_argument(
        "--dataset",
        choices=["ava", "ami", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only generate the download manifest",
    )
    args = parser.parse_args()

    if args.manifest_only:
        generate_download_manifest()
        return

    print("=" * 70, flush=True)
    print("   Dataset Download for Video-Restoration Experiments", flush=True)
    print("=" * 70, flush=True)

    results = {}
    if args.dataset in ("ami", "all"):
        results["ami"] = download_ami_meetings()

    if args.dataset in ("ava", "all"):
        results["ava"] = download_ava_videos()

    print("\n  Generating final download manifest with SHA-256 hashes...", flush=True)
    generate_download_manifest()

    print("\n" + "=" * 70, flush=True)
    print("FINAL STATUS", flush=True)
    print("=" * 70, flush=True)
    if "ami" in results:
        r = results["ami"]
        print(f"  AMI: {len(r['downloaded'])}/15 meetings complete ({len(r['partial'])} partial, {len(r['failed'])} failed)", flush=True)
    if "ava" in results:
        r = results["ava"]
        print(f"  AVA: {len(r['downloaded']) + len(r['skipped'])}/25 videos complete ({len(r['failed'])} failed)", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()

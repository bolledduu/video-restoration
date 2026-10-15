# Reproduce the CPU continuation

For the latest **LL-Bench image-selection pilot**, use the lightweight commands in [LLBENCH_RESULTS.md](LLBENCH_RESULTS.md#reproduce-and-reuse). That phase needs only Python/NumPy and public CSVs; none of the video/model setup below is required.

Run from the repository directory. Tested on Windows, Python 3.13.0; six CPU cores, ~7.4 GiB usable RAM. The scripts set two inference threads and seed 0. Allow several minutes for six FastDVDnet excerpts. Media and model weights stay in Git-ignored directories.

## Dependencies

Use an isolated environment if installing fresh. The actual run reused installed CPU packages; only the two small audio/download utilities were added in `tools/python`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu-lock.txt
```

`requirements-cpu-lock.txt` pins direct dependencies; transitive dependencies are not completely locked. PyAV bundles the required codecs. FFmpeg/ffprobe CLI is necessary only for legacy `pilot.py` actions, not the commands below.

## Official model code and weights

```powershell
git clone https://github.com/m-tassano/fastdvdnet.git external/fastdvdnet
git -C external/fastdvdnet checkout c8fdf6182a0340e89dd18f5df25b47337cbede6f
git clone https://github.com/TaoRuijie/TalkNet-ASD.git external/TalkNet-ASD
git -C external/TalkNet-ASD checkout 6d6821479af485e251c4991487e40573b42181b4
python -c "import gdown; gdown.download(id='1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea', output='external/TalkNet-ASD/pretrain_TalkSet.model', quiet=False)"
```

FastDVDnet's `model.pth` is included in its upstream repository. TalkNet's Drive ID comes from official `demoTalkNet.py`. Both wrappers load state dictionaries strictly with `weights_only=True`, on CPU. Exact checkpoint SHA-256 values are in `results/video_baseline.json` and `results/talknet_diagnostics.json`. Neither script imports training/data-download entry points or changes the upstream source.

## Ordered execution

```powershell
python -m unittest discover -s tests -v
python scripts/fetch_sources.py
python scripts/cpu_media.py
python scripts/video_baseline.py
python scripts/talknet_cpu.py
python scripts/build_report.py
python scripts/finalize_cpu.py
```

Run sequentially: later scripts require all prior outputs. `fetch_sources.py` uses the dataset commit pinned in `manifest.json`, checks exact metadata and records hashes. Review licensing in the downloaded card. `cpu_media.py` creates the six full-clip degradations. `video_baseline.py` uses ~2-second excerpts and produces reference/degraded/restored arrays plus silent previews. `talknet_cpu.py` uses the exact arrays, fixed source audio, provisional face proposals and three separate modality outputs. `build_report.py` requires all 18 diagnostics, validates audio invariance and creates local `data/review/index.html`.

The scripts overwrite their own derived outputs/results when rerun. Raw clips are reproducibly re-downloaded by fetch. Checkpoint/code revisions should be verified against the pinned commands before comparing runs. Local package-version and runtime metadata in committed reports describe this measured run, not an automatically detected future machine.

Optional availability refresh:

```powershell
python scripts/model_search_audit.py
```

## Meaning of the outputs

- `results/source_downloads.json`: pinned metadata, source URLs, byte sizes and hashes.
- `results/provisional_annotations.json`: visual observations, decoded audio signal statistics, explicitly unverified labels/timing.
- `results/cpu_degradations.json`: complete stream/frame/timestamp audits and all-frame metrics.
- `results/video_baseline.json`, `quality_summary.csv`: genuine video-only inference with provenance and measured quality.
- `results/talknet_diagnostics.json`: raw logits, timestamps, face boxes, audio hashes, model/code hashes. No verified accuracy.
- `results/research_status.json`: NO-GO full-paper investment decision, closed feasibility assessment, and null unavailable metrics.

No C3/C4 result can be produced without an actual audio-conditioned restoration release. No runnable GPU configuration is claimed for missing code/weights. No lip-sync model is substituted. To continue the science, first secure that release and independently annotate suitable face tracks; keep conditioning audio and downstream audio separate in every new condition record.

## Supplemental Hugging Face output evaluation

See `HF_CONTINUATION.md` for the three public OmniVR triplets, exact provenance and ordered CPU commands. This is author-output evaluation, not local OmniVR restoration.

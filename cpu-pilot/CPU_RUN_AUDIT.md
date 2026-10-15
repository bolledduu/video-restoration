# Laptop execution audit

Date: 2026-10-12, America/Los_Angeles. Starting branch: `wvaq-cpu-pilot`, commit `4883c980a2d3526285ee3f3e3d0744b9215b4aa7`. PR #1 was already merged; continuation branch: `codex/cpu-experiment-audit`.

## Environment checked

- Windows 11 build 10.0.26300; Python 3.13.0.
- Ryzen 5 5600H, 6 physical / 12 logical cores.
- Initial visible RAM 7,764,856 KiB (~7.41 GiB), free 1,225,820 KiB (~1.17 GiB); availability changes with usage.
- PyTorch 2.8.0+cpu; CUDA unavailable.
- FFmpeg/ffprobe CLI absent from PATH. Used installed PyAV 16.0.1 and bundled FFmpeg libraries; exact versions in `results/media_environment.json`.
- OpenCV 4.12.0, NumPy 2.2.6, SciPy 1.16.2, scikit-image 0.25.2, Pillow 12.1.0. Added only small `python_speech_features==0.6` and `gdown==5.2.0` packages in ignored `tools/python/`.

## Reproduction and fixes

Inspected all original tracked files, status and branches before editing. Five original tests passed outside the sandbox. The initial sandbox run failed in two tests due to temporary-directory permissions; DNS/hardware queries were similarly blocked and succeeded with approved outside-sandbox execution. These are environment restrictions, not scientific failures.

Corrected automatic annotation validity, silent truncated-video comparisons, default 120-frame-only measurement, tests overwriting persisted results, and misleading labeling of multiple-choice errors as attribution measurements. Added Git exclusions for source/derived media, model weights and external code.

New PyAV processing streams frames, copies authentic audio packets, and verifies timing/counts. Eight tests pass, including truncated-pair rejection, identical-video metrics and preventing annotation verification based on a file path.

## Executed outputs

Three real source clips; six full degraded clips; six FastDVDnet excerpts; 18 TalkNet condition diagnostics with three modality heads. See `results/` for hashes, parameters, measurements and null scientific effect estimates. Media remains local under `data/`.

FastDVDnet inference took 48.71–60.57 s per excerpt. Final TalkNet inference took ~1.02–1.60 s per condition, excluding import/loading/preprocessing. These are measured local times, not general performance claims.

## Unresolved

No official executable audio-conditioned restoration/checkpoint located in the bounded search; this does not demonstrate CPU incompatibility. Speaking labels remain provisional by user request. C3/C4, verified natural C5, false-attribution difference and accuracy remain unavailable. A runnable GPU configuration cannot be truthfully supplied without the model release. Earlier unrelated-fixture smoke metrics are excluded from all current research tables.

## Final laptop-only assessment

The final offline paired analysis validates all six condition groups and summarizes saved model outputs in `results/cpu_paired_diagnostics.csv`. Visual-only decisions change in 5/6 cases and fixed-audio joint decisions in 4/6; audio-only outputs are invariant. These are prediction disagreements, not verified errors. `FINAL_CPU_DECISION.md` closes the feasibility assessment with NO to full-paper investment. No cloud node was requested.

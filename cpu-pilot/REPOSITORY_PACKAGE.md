# Repository package and local assets

This package contains the complete Git-visible project snapshot: code, protocols, tests, historical and latest results, research questions, reports, plots, and Qwen tool traces. The former root-level simulation/demo implementation was removed from the active tree on 17 October 2026; it remains in Git history at commit `3fbc89e`.

The [current research decision](RESEARCH_DECISION.md) separates successful workflow validation, observed benefits and harms, and unsuccessful safeguard screens. No validated novel framework or paper-readiness claim is established. Read the [latest AVA results](results/ava_smoke_v1/RESULTS.md); older reports retain their original scopes.

## Intentionally local assets

Downloaded third-party datasets, generated media arrays, external repositories, model checkpoints, installed Python packages, credentials, caches and `.git` internals are not uploaded. Existing ignore rules exclude `data/`, `external/` and `tools/`. These assets are large, have their own redistribution terms, or contain machine-specific state. Recorded URLs, pinned revisions and hashes identify the inputs used.

`reproducibility/` preserves input manifests and a SHA-256 inventory of the imported project files. Paths inside historical result JSONs record the original laptop location; they are provenance records, not portable paths to files hosted by GitHub. Audit scripts that require local arrays must run after downloading/preparing those arrays and adapting stored paths to the new checkout.

## Reproduction entry points

Run commands from the `cpu-pilot/` directory. See [REPRODUCE.md](REPRODUCE.md) and the individual experiment reports for the historical sequence. Reproduction on a fresh machine was not executed as part of this upload.

The newest pilot additionally uses `pyiqa==0.1.14.1`, `opencv-contrib-python-headless==4.12.0.88` (for `cv2.quality`), and `requests`. The original run used isolated local dependency directories to avoid incompatible OpenCV wheels; do not install overlapping OpenCV packages indiscriminately. Python 3.13 and Torch 2.8 CPU were used. Direct earlier dependencies are in `requirements-cpu-lock.txt`; it is not a complete transitive environment lock.

Models and sources:

- FastDVDnet: https://github.com/m-tassano/fastdvdnet at `c8fdf6182a0340e89dd18f5df25b47337cbede6f`; newest pilot uses included `model_clipped_noise.pth`.
- TalkNet: https://github.com/TaoRuijie/TalkNet-ASD at `6d6821479af485e251c4991487e40573b42181b4`; checkpoint download instructions are in REPRODUCE.md.
- UniTalk-ASD: https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD at `0c0f80331c91b57ce772bc674f81d40a24247383`; source paths and exclusions are in the saved manifests. Run historical preparation before `prepare_agent_pilot.py`, which reads earlier manifests to exclude previously scored sources.
- AMI annotations: https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip â†’ `data/agent_pilot/ami/annotations.zip`.
- AMI video: https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/video/ES2002a.Closeup1.avi â†’ `data/agent_pilot/ami/closeup1.avi`.
- AMI audio: https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav â†’ `data/agent_pilot/ami/mix.wav`.
- BRISQUE assets and MUSIQ setup are documented in WVAQ_INITIAL_HYPOTHESIS.md and WVAQ_FAILURE_DISCOVERY.md. Weights and input hashes are saved in result files.

Newest run order, once those assets and earlier manifests exist:

```text
python scripts/prepare_agent_pilot.py
python scripts/agent_pilot_measure.py
python scripts/ami_agent_pilot.py
python scripts/qwen_evidence_agent.py
python scripts/audit_agent_pilot.py
python scripts/report_agent_pilot.py
```

Qwen reads `GROQ_API_KEY` from the environment. Never add a key to code or a report. `GROQ_QWEN_MODEL` can select an available Qwen model; the runner verifies account availability and makes no non-Qwen substitution. Re-running invokes a paid/rate-limited external API according to your account terms. Saved results require no API call to read.

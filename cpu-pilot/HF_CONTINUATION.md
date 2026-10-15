# Hugging Face continuation: CPU evaluation of OmniVR author outputs

The broader search found public joint audio-video restoration weights and outputs for [OmniVR](https://huggingface.co/xin1u/OmniVR). This corrects the earlier overly broad availability conclusion. GAVN/DAVD-Net were not the only possible models.

**Executed here:** download and hash verification of nine published files; CPU FastDVDnet inference on three additional excerpts; CPU TalkNet inference on twelve conditions; paired fidelity and prediction-change measurements. **Not executed here:** OmniVR restoration inference or conditioning-audio swaps. The original causal hypothesis remains untested; this is a supplemental output audit, not its completion.

## Sources and compute

Pinned HF revision: `3a58cc2cf24262ce7b81e053454c462614fd5eae`. The first three `with_gt` records in filename order were selected before examining outputs. The matched `predictions/withgt_step02400_neg_cfg` files are author-generated multistep outputs, not the newer Flash model. The file associations and checkpoint identity follow the release; we did not independently reproduce their generation. The published benchmark includes both audio and visual degradation.

The ordinary OmniVR path loads a 46,149,344,974-byte LTX checkpoint plus a 7,605,551,032-byte adapter and other components. The inspected upstream multistep entry point selects CUDA. This laptop has about 7.4 GiB usable RAM. Shorter clips do not reduce checkpoint weight storage. No compatible small CPU implementation was established, so these weights were not downloaded and no claim of an inference attempt is made. The public Flash path lacks required compatible artifacts according to its model card. `results/omnivr_execution_feasibility.json` records the source revision and code evidence.

No authentication was needed, no supplied token was used or saved, and no paid compute was requested. Media stays Git-ignored. Public availability is not a finding about all underlying source-media rights.

## Protocol

All three downloaded streams per sample were fully decoded to check frame counts and timestamps. Quality evaluation uses clip-relative seconds 1–3 resized identically to 256×144, with no output alignment optimized against the reference. These measurements do not reproduce the paper's resolution or metrics. FastDVDnet uses the existing noise map 10/255 without added noise: an explicitly task-mismatched baseline, not an OmniVR ablation.

Reference-derived face boxes are reused across conditions. TalkNet uses nearest-frame resampling to 25 Hz and the clean reference soundtrack for every condition, never OmniVR's restored soundtrack. Audio-only logits must be identical. Visual-only inference precedes cross-attention. Logits are not calibrated confidence. Largest-face crops are not independently verified identity tracks; no accuracy, false-attribution rate or new speaking labels are fabricated.

## Measured results

PSNR in dB; changed frames compare OmniVR versus the degraded input. These are prediction disagreements, not errors. All values refer to short resized excerpts.

| Sample | Degraded PSNR | FastDVDnet PSNR | OmniVR PSNR | Visual-only changed | Fixed-audio AV changed |
|---|---:|---:|---:|---:|---:|
| speech_000 | 13.906 | 13.983 | 13.944 | 22/49 | 1/49 |
| speech_001 | 20.969 | 21.018 | 22.331 | 5/49 | 9/49 |
| speech_002 | 16.713 | 16.950 | 19.739 | 3/49 | 0/49 |

RGB SSIM, all modality logits, decoded stream metadata, crop boxes, code revisions, hashes and per-condition counts are in `results/omnivr_cpu_evaluation.json`; the compact table is `results/omnivr_summary.csv`. Local comparisons and silent previews are in `data/omnivr/evaluation/`.

All nine source files decode to 121 frames. Source/reference dimensions are 1920x1080; author outputs are 1920x1088. All are resized to the evaluation grid, so the small aspect-ratio difference and generated geometry are additional limitations for pixel and crop comparisons.

Sampled-frame inspection: speech_000 changes the green table to gray, alters background texture and subtitle content; speech_001 changes shirt color and ceiling appearance; speech_002 changes some colors and background detail. The low-quality inputs have lost color and other detail. Reconstruction differences can therefore reflect lost visual information and generative priors; no observation identifies audio as their cause. These sampled comparisons are not full-motion human annotations or a novel hallucination claim.

## What this establishes

We now have local measurements on genuine joint-restoration outputs, with downstream audio held fixed. This is progress beyond video-only restoration, but cannot isolate audio conditioning: the restorer changes pixels for multiple reasons, and there is no conditioning-only intervention. The clean reference is an image-quality target, not a frame-level speaker annotation. Caption text, speech ratio and model predictions are not substituted for labels.

**Full-paper investment remains NO on current evidence.** Model availability is no longer a blanket blocker; the outstanding requirements are executable inference within the available resources, counterfactual outputs and valid speaker labels. These three preselected clips do not establish a population effect or a publication claim. The original AV-SpeakerBench results are retained separately and are not relabeled as OmniVR results.

## Reproduce

With the already documented CPU dependencies and official FastDVDnet/TalkNet checkpoints:

```powershell
python scripts/fetch_omnivr.py
python scripts/evaluate_omnivr_cpu.py
python scripts/summarize_omnivr.py
```

These commands require no Hugging Face login. The first downloads only the nine selected media files and their public metadata. The second executes CPU baseline and detector inference; it does not load OmniVR weights.

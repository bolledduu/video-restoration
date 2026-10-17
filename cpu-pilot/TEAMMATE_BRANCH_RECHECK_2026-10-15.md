# Recheck of the teammate's evidence branch

The stored results support neither “all predictions are the same” nor “the research hypothesis has been disproved.” A specific restoration-induced detector error from the labelled pilot is reproducible. The broader hypothesis was not proved by that pilot.

Audited branch: [`results/final-evidence-2026-10-15`](https://github.com/bolledduu/agent-audio-visual/tree/5f8307aee5ffb0f83c7f17ee343a23bd9626d4a3), pinned commit `5f8307aee5ffb0f83c7f17ee343a23bd9626d4a3`. Its parent is the October 13 preparation commit `e3305eb`, so it does not include the later UniTalk/Qwen agent pilot. The two experiments must be compared explicitly.

## 1. Same-data restoration rerun: completed

Executed the branch's unmodified FastDVDnet and TalkNet scripts on the three matching AV-SpeakerBench source files. All three source SHA-256 hashes match the branch. Repeated all six restoration conditions and all 18 downstream condition runs. The original audio, crop boxes, timestamps, checkpoint hashes, and decision threshold are unchanged. Restored outputs were recomputed, not replaced with cached predictions.

**Every saved branch decision was reproduced.** Floating-point logits differ by at most 0.000355 across the recorded environments, without changing any class decision. Input/output quality measurements match within 0.00001. Audio-only outputs remain exactly invariant across visual conditions within the new run.

| Source video | Moderate: changed frame decisions | Severe: changed frame decisions |
|---|---:|---:|
| S1 — `I0CPeFpSIbE`, clip-relative 1.0–3.0 s | 5/49 | 2/49 |
| S2 — `V9WCCw5UWmw`, clip-relative 6.0–8.0 s | 30/47 | 5/47 |
| S3 — `gFNigX2jrnM`, clip-relative 6.5–8.5 s | 0/49 | 0/49 |

The denominator is the aligned TalkNet frame count. Each row compares degraded video plus original audio against FastDVDnet-restored video plus the same audio. Across the six cases, **42/290 frame decisions change in 4/6 cases**. There are 37 speaking-to-silent changes and five silent-to-speaking changes. These are dependent frame observations, not 290 independent samples.

The source clips have provisional speaker annotations and a largest-face crop heuristic rather than verified identity tracks. Consequently, **the 42 changes cannot be called errors or corrections**. S2 severe's five silent-to-speaking changes could be corrections. Reference-model predictions are not ground truth. The branch itself records this limitation.

FastDVDnet here uses the original `model.pth` and a fixed 10/255 noise map on blur/compression inputs with no added Gaussian noise. This is the historical, task-mismatched baseline. Its unchanged S3 outcomes do not invalidate the separate, task-matched Gaussian denoising pilot.

## 2. What the new Qwen records actually contain

The two entries in `qwen_groq_batch.json` are **different DMD driver videos**, identified by different hashes. Both return `No, 0.95`. That is not a before/after pair.

The committed batch script samples the first image from each video. Its default prompt is:

> Look at this face image. Is the driver likely looking forward? Reply in one short sentence with yes/no and a confidence score.

No audio or restoration is supplied by that batch script. One saved probe response explicitly describes the driver's gaze direction. Neither results file records a restored-condition hash, a restoration checkpoint, paired before/after predictions, or task labels. The saved records also omit the full executed prompt/request, so an environment-variable prompt override cannot be reconstructed.

The exact two DMD media files are absent from this laptop and from the Git branch; the recorded locations are on the teammate's Linux machine. **Those API calls were not rerun.** The completed same-data rerun above covers the branch's reproducible restoration experiment. The DMD entries cannot establish restoration safety or restoration harm, and cannot test audiovisual speaker attribution.

## 3. Event-counter audit

`scripts/false_restoration_event.py` has three measurement problems:

1. It calls any prediction transition from 0 to 1 a false positive without checking a true speaking label. On a speaking person, that transition is a correction.
2. Its `event_rate` denominator is the number of event frames, rather than all valid paired frames. Therefore any nonzero result has rate 1.0.
3. Rows without clip/frame keys are silently discarded. Passing the committed Qwen batch records to it returns zero events after comparing **zero valid pairs**.

Executed diagnostic fixture: ten complete pairs, one 0-to-1 change, true label 1. The branch counter returns one “false-positive” event and rate 1.0. The actual prediction-flip rate is 0.1 and the actual false-positive count is zero. This fixture tests software semantics; it is not research data. No committed event-counter output was found that establishes the teammate used this function for the quoted conclusion.

The rerun report uses the raw paired logits, reports both directions of change, and leaves accuracy undefined when labels are unavailable.

## 4. Our labelled FastDVDnet pilot: reproduced separately

Recomputed all eight existing cases using official clipped-noise FastDVDnet weights and the true noise sigma. Reconstructed arrays match the earlier saved arrays byte for byte; TalkNet scores match within 0.000001. Audio is fixed across conditions. These are two source videos and four one-second tracks, not eight independent sources.

| Source | Dataset label | Noise sigma | Captured errors / 24 | Restored errors / 24 | New errors | Corrections |
|---|---|---:|---:|---:|---:|---:|
| `kgVz_AXYvn0` | silent | 15 | 8 | 3 | 0 | 5 |
| `kgVz_AXYvn0` | silent | 40 | 3 | 0 | 0 | 3 |
| `kgVz_AXYvn0` | speaking | 15 | 24 | 24 | 0 | 0 |
| `kgVz_AXYvn0` | speaking | 40 | 0 | 24 | 24 | 0 |
| `EEey6Pt0rqI` | silent | 15 | 0 | 0 | 0 | 0 |
| `EEey6Pt0rqI` | silent | 40 | 0 | 5 | 5 | 0 |
| `EEey6Pt0rqI` | speaking | 15 | 0 | 0 | 0 | 0 |
| `EEey6Pt0rqI` | speaking | 40 | 0 | 0 | 0 | 0 |

Total: 35/192 errors before restoration and 56/192 afterward, with 29 new errors and eight corrections. **Do not treat all 29 new errors as invented evidence.** Twenty-four occur on a speaking track that TalkNet also gets wrong on the source reference. Noise had accidentally made its decisions correct; restoration returns it to a detector-failure regime.

The cleaner counterexample is the dataset-labelled silent track `EEey6Pt0rqI:2`, beginning at source time 7.28 s, sigma 40. Captured and source-reference predictions are correct on all 24 scored frames, while restored video causes five false speaking predictions. SSIM improves from 0.1166 to 0.9272. Previously measured MUSIQ improves from 13.74 to 15.96, and BRISQUE improves from 62.78 to 61.10 (lower is better). SSIM was recomputed; the two no-reference metrics were carried forward only after identical output arrays were verified.

These five errors are at indices 0–4, the start of a one-second crop. Their speaking scores range from 0.526 to 0.600 and are uncalibrated. Boundary effects and detector sensitivity remain plausible explanations. No invented lip motion, human perceptual preference, calibrated high confidence, or wrong person–quotation attribution is established.

Both quality scores improving is also different from both assessors selecting the same best candidate. At sigma 40, BRISQUE selects NLM and MUSIQ selects FastDVDnet. The earlier harmful consensus-selection example uses NLM at sigma 15 and has two new errors. The predefined requirement for harmful consensus selection on at least two sources still **fails**: only one source qualifies.

Qwen was a separate tool-using agent receiving measurements. It did not make these TalkNet frame decisions. Its previously saved results must not be relabelled as direct video/audio understanding or as new Qwen failures.

## 5. Additional-video check

Results are recorded separately in `results/additional_fastdvdnet_videos.json`. This run applies the same clipped-noise FastDVDnet checkpoint to all three preselected tracks from the earlier NLM transfer probe. Their saved noisy arrays and sigma 25 are fixed; no new degradation search is performed. All three tracks are dataset-labelled silent, so the check cannot establish performance on speaking tracks. They are additional to the two-source FastDVDnet pilot, but were already examined with NLM.

**Completed: no new errors on any of the three additional videos.**

| Source | Dataset label | Captured errors / 24 | FastDVDnet errors / 24 | New errors | Corrections |
|---|---|---:|---:|---:|---:|
| `9Zja_cIAwY8` | silent | 0 | 0 | 0 | 0 |
| `ABqo634gI5s` | silent | 0 | 0 | 0 | 0 |
| `qv3-HaaxGUc` | silent | 0 | 0 | 0 | 0 |

All 72 scored frames remain correct under the dataset labels. The source-reference decisions are also correct. Captured/reference scores reproduce the previous NLM probe's measurements within 0.000001, and audio-only scores are identical across conditions. The new restored arrays and side-by-side previews are saved under `data/additional_fastdvdnet_videos/`. This check **does not replicate the harmful effect** and must be retained as a negative result for these fixed settings.

## 6. Scientific conclusion

**A reproducible, condition-specific downstream detection failure exists. The broader hypothesis was not proved, and the teammate's stored results do not disprove it.** A no-change result is evidence about that tested condition. It does not refute an existence claim that restoration can sometimes introduce errors.

The evidence supports studying when restoration preserves speaker information. It does not certify a novel safeguard, a successful full-paper method, or audio-conditioned false corroboration. FastDVDnet has no conditioning-audio input. A positive detector example and negative transfer cases should both remain in the research record.

## Reproduction and evidence

- [Branch replay and audit JSON](results/teammate_branch_reproduction.json)
- [Paired branch counts CSV](results/teammate_branch_reproduction.csv)
- [Labelled pilot replay JSON](results/restoration_before_after_reproduction.json)
- [All eight labelled comparison videos](data/restoration_comparison/index.html)
- [Labelled score traces](results/restoration_before_after_traces.png)
- [Additional-video results](results/additional_fastdvdnet_videos.json)
- [Three additional comparison videos](data/additional_fastdvdnet_videos/index.html)

From the `video-restoration` directory, with the existing Python CPU environment and original runtime checkout:

```powershell
python cpu-pilot/scripts/reproduce_teammate_branch.py --runtime-root "C:\Users\bolle\Downloads\cvpr submission\agent-audio-visual"
python cpu-pilot/scripts/check_additional_fastdvdnet_videos.py --runtime-root "C:\Users\bolle\Downloads\cvpr submission\agent-audio-visual"
```

The branch replay stages exact scripts under ignored local data, reads existing model/tool directories through directory junctions, and saves new outputs separately. The original checkout and original result files are preserved. No paid GPU or new API requests were used.

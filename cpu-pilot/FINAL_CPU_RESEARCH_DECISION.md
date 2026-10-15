# Completed CPU research pilot

**Decision: REJECT scaling this tested setup.**

Changing restoration audio introduced 25 wrong frame decisions and corrected 45. No frames met the joint quality-improvement and higher-scored-error endpoint. Restoration reduced average reference fidelity at both degradation levels.

At 75% retained coverage, captured-video agreement reduced error from 55.56% with confidence selection to 45.83%. The safeguard screen passed; the combined research gate failed because RQ1 failed.

This decision comes from executed, labeled speaker-attribution experiments. It is not a verdict inferred from missing GPU access or missing annotations. It applies to the tested framewise Ear-for-Face-SR + TalkNet setup; it does not prove or disprove every audio-conditioned restoration architecture.

## Final operational research questions

1. **False corroboration:** Can changing conditioning audio cause framewise face restoration to improve reference-based image quality while making a downstream active-speaker classifier wrong with a higher wrong-class score?
2. **Evidence origin:** How does loss of captured visual information change audio-caused speaker-decision errors when the visual input and downstream audio are held fixed?
3. **Independent evidence assessment:** Can audio-intervention sensitivity, agreement with captured-video predictions, or re-degradation consistency reduce speaker errors at the same retained coverage as confidence-based selection?

These operationalize the original topics for a CPU-feasible, framewise restorer. They do not silently treat a still-image network as a trained temporal model. “Score” means uncalibrated softmax, not calibrated probability of correctness.

## What actually ran

- 8 one-second clips from 4 UniTalk validation source videos, balanced four speaking/four non-speaking; labels were supplied by the dataset authors.
- Two degradation levels (8×8 and effective 4×4), three restoration conditions (visual-only, original soundtrack, different-source soundtrack), and first-frame full-path repeats: **1216 completed, retained generator forward passes on CPU**.
- Every one of the 25 target frames is restored independently. The restorer receives the same four-second audio context throughout each target clip. This does not test time-varying conditioning or learned temporal restoration.
- TalkNet evaluates reference, degraded, visual-only, matched, and swapped videos. Its original one-second soundtrack remains unchanged. All audio-only invariance checks and all restoration repeat checks must pass.
- MFCC framing provides 24 scored frames per one-second clip; the final unaligned frame is excluded consistently rather than padded. Frames and degradation variants are not treated as independent videos.
- Selection and binary decision criteria were committed before restoration inference in `b34ee56`. No cases were selected by prediction or restoration outcomes.

Execution scheduling was changed after four completed combinations to group encoding before synthesis and reduce memory paging. Those four outputs were retained after checksum checks; an unfinished combination was discarded and restarted. No model operation, sample, condition, or decision criterion was changed. The retained-pass count excludes discarded partial work and initialization.

## Measured results

| Case | Input size | Label | Reference error | Degraded error | Visual-only error | Matched error | Swapped error | Joint harmful frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lnAa1N5ynyI__label0 | 8 | 0 | 0.0% | 0.0% | 100.0% | 29.2% | 100.0% | 0 |
| lnAa1N5ynyI__label0 | 4 | 0 | 0.0% | 0.0% | 20.8% | 0.0% | 0.0% | 0 |
| lnAa1N5ynyI__label1 | 8 | 1 | 66.7% | 41.7% | 45.8% | 100.0% | 75.0% | 0 |
| lnAa1N5ynyI__label1 | 4 | 1 | 66.7% | 41.7% | 100.0% | 95.8% | 100.0% | 0 |
| Pc5iQdNpDX0__label0 | 8 | 0 | 0.0% | 0.0% | 25.0% | 12.5% | 0.0% | 0 |
| Pc5iQdNpDX0__label0 | 4 | 0 | 0.0% | 4.2% | 0.0% | 0.0% | 4.2% | 0 |
| Pc5iQdNpDX0__label1 | 8 | 1 | 0.0% | 20.8% | 83.3% | 83.3% | 0.0% | 0 |
| Pc5iQdNpDX0__label1 | 4 | 1 | 0.0% | 75.0% | 91.7% | 91.7% | 87.5% | 0 |
| QRXbRSq8p4s__label0 | 8 | 0 | 0.0% | 0.0% | 91.7% | 79.2% | 79.2% | 0 |
| QRXbRSq8p4s__label0 | 4 | 0 | 0.0% | 41.7% | 0.0% | 58.3% | 8.3% | 0 |
| QRXbRSq8p4s__label1 | 8 | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0 |
| QRXbRSq8p4s__label1 | 4 | 1 | 100.0% | 100.0% | 29.2% | 37.5% | 29.2% | 0 |
| ho_oK12hY6M__label0 | 8 | 0 | 0.0% | 4.2% | 95.8% | 75.0% | 75.0% | 0 |
| ho_oK12hY6M__label0 | 4 | 0 | 0.0% | 20.8% | 0.0% | 0.0% | 0.0% | 0 |
| ho_oK12hY6M__label1 | 8 | 1 | 54.2% | 100.0% | 100.0% | 79.2% | 100.0% | 0 |
| ho_oK12hY6M__label1 | 4 | 1 | 54.2% | 95.8% | 100.0% | 100.0% | 100.0% | 0 |

| Input size | Condition | Mean SSIM | Mean PSNR | AV error | Wrong-class score ≥0.9 frames |
|---|---|---:|---:|---:|---:|
| 8 | degraded | 0.6032 | 19.743 | 33.33% | 0 |
| 8 | visual_only | 0.4592 | 17.211 | 80.21% | 4 |
| 8 | matching | 0.4598 | 17.271 | 69.79% | 0 |
| 8 | swapped | 0.4582 | 17.209 | 66.15% | 0 |
| 4 | degraded | 0.5180 | 17.257 | 47.40% | 0 |
| 4 | visual_only | 0.4417 | 16.330 | 42.71% | 0 |
| 4 | matching | 0.4392 | 16.317 | 47.92% | 6 |
| 4 | swapped | 0.4395 | 16.362 | 41.15% | 0 |

The ≥0.9 column is a descriptive model-score threshold, not a claim of calibrated 90% confidence. The declared joint event below, not this secondary summary, determines the pilot decision.

A joint harmful frame requires the reference, degraded, and matching-audio predictions to be correct; the swapped-audio prediction to be wrong; wrong-class score to rise by at least 0.10; and swapped restoration SSIM to exceed degraded-video SSIM. At least five such frames in each of two independent source videos are required for the positive RQ1 pilot screen.

**RQ1 screen: FAIL.** Qualifying source videos: 0 of 4.

| Input size | Mean audio-induced pixel change | Correct→wrong frames from swap | Wrong→correct frames from swap | Joint harmful frames |
|---|---:|---:|---:|---:|
| 8 | 5.349 / 255 | 23 | 30 | 0 |
| 4 | 3.766 / 255 | 2 | 15 | 0 |

| Secondary visual-only head | Correct→wrong from audio swap | Wrong→correct from audio swap | Joint harmful frames |
|---|---:|---:|---:|
| 8 | 0 | 6 | 0 |
| 4 | 2 | 3 | 0 |

The visual-only head receives no direct audio: its intervention response can only come through restored frames. This is a secondary endpoint; the predeclared AV gate is not replaced after observing outcomes.

RQ2 reports a controlled audio effect on pixels and labeled decisions at two degradation levels. It does not infer a specific invented anatomical detail from a classifier error or establish formal non-identifiability.

| Safeguard | Error at 50% coverage | Error at 75% coverage |
|---|---:|---:|
| confidence | 64.06% | 55.56% |
| conditioning_sensitivity | 54.17% | 54.17% |
| capture_disagreement | 49.48% | 45.83% |
| reprojection_residual | 43.23% | 47.22% |

**RQ3 safeguard screen: PASS.** A pass requires lower pooled error than confidence at both coverages and lower error at 75% coverage in at least two source videos. These are fixed, untuned baseline rules, not a newly trained or generally validated safeguard.

## Interpretation and limits

The positive screen for scaling requires both RQ1 and RQ3. A failure is a negative result for that predeclared screen on this setup, not proof that the broad scientific hypothesis is false. There is no remaining execution step in this CPU pilot.

The sample is deliberately small and selected from the smallest eligible validation archives. Four source videos cannot establish population-wide effect sizes or publication readiness. All cases, including reference-model errors, are retained. Published labels are used as benchmark ground truth; no claim of independent human relabeling is made.

The released image restorer is adapted through TensorFlow compatibility mode and applied framewise. It was designed for face/voice identity cues, not temporal lip synchronization. Crops have at least 64 pixels per stored side, but their original capture resolution is not independently known. Effective 4×4 is outside the model’s native 8×8 input distribution. SSIM/PSNR measure reference fidelity, not human perceptual quality.

Audio swaps replace the restoration soundtrack only. “Matching” means the original scene soundtrack, which can contain another person speaking when the target face is silent. Original labels and original downstream audio stay fixed. This is precisely why negative face tracks are included.

## Reproduce

```powershell
python scripts/prepare_unitalk_cpu.py
python scripts/audit_unitalk_inputs.py
tools/ear_cpu/Scripts/python.exe scripts/ear_cpu_probe.py --unitalk
python scripts/evaluate_unitalk_cpu.py
python scripts/audit_unitalk_results.py
python scripts/report_unitalk_cpu.py
```

Use the existing isolated Ear environment and pinned weights described in ORIGINAL_RQ_PREPARATION.md. Raw media and restored faces remain local. Re-running preparation recreates input archives and hashes; the inference runner verifies cached completed outputs before skipping them.

Sources: [UniTalk dataset](https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD/tree/0c0f80331c91b57ce772bc674f81d40a24247383), [dataset authors’ code](https://github.com/plnguyen2908/UniTalk-ASD-code), [Ear-for-Face-SR](https://github.com/gmeishvili/ear_for_face_super_resolution/tree/c7f342b1b2984ed674b7407a95305f9cd6f89613), [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD/tree/6d6821479af485e251c4991487e40573b42181b4).

Independent arithmetic verification: results/unitalk_result_audit.json (separate standard-library calculation of saved scores; no independent relabeling).

Machine-readable results: results/unitalk_cpu_evaluation.json. Full frame scores: results/unitalk_cpu_frame_scores.json. Selection/label provenance: results/unitalk_cpu_inputs.json and results/unitalk_input_audit.json.

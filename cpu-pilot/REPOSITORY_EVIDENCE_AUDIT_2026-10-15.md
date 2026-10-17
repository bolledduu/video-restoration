# Repository evidence audit — 15 October 2026

This audit supports method design for the current three research questions. It does not replace saved experimental results, change historical screening decisions, or report a new model experiment.

The GitHub default-branch head was checked through the GitHub connector and matched the local checkout: `08d9ff2a21e88a38da379e1193a75c228a4dff44`. The separate root implementation was introduced in `37779385fff9b21f1e19a9a6f49ceb05fb2295cc`. Its experiments and the imported `cpu-pilot/` experiments must remain separate.

## 1. What yesterday's root-level agent experiments establish

### Real-video log

Inspected `pure_real_video_qwen_experiment.py` and recomputed counts directly from `results/pure_real_video_qwen_log.json`.

- Four windows from one video, each with three degradation conditions: 12 window-condition cases, not 12 independent recordings.
- Three candidates: captured, bicubic resize, unsharp masking. There is no learned generative-restoration checkpoint in this runner.
- All three candidates have the same logged attribution in all 12 cases: 10 match the assigned label and two abstain. No candidate produces a wrong named-person answer; there are no correct-to-incorrect transitions or corrections.
- Both quality-only and gated selection choose unsharp masking in all 12 cases. Both finish with 10/12 correct and two abstentions, exactly like captured input.
- Per degradation, the captured counts are 3/4 correct for clean, 4/4 for moderate, and 3/4 for severe. The saved real-video log therefore does not support the README's claimed collapse to 0% accuracy under severe degradation.
- Qwen receives text containing pixel-difference measurements, a sharpness score, and a rule-generated speaker verdict. The request contains no video, images, or waveform. The runner does not analyze the recorded audio.
- Target statements and speaker labels are hard-coded in `REAL_WINDOWS`; a verified transcript/annotation provenance path is not established by this runner. Treat its correctness as agreement with those assigned labels until independently verified.
- The quantity called mouth motion is mean absolute grayscale intensity difference. It is not measured optical flow or displacement in pixels/frame.

This is useful as an integration exercise with real pixels and logged Qwen responses. It is not confirmation of real generative-restoration-induced speaker misattribution.

Evidence: [runner at reviewed commit](https://github.com/bolledduu/video-restoration/blob/08d9ff2a21e88a38da379e1193a75c228a4dff44/pure_real_video_qwen_experiment.py), [saved log](https://github.com/bolledduu/video-restoration/blob/08d9ff2a21e88a38da379e1193a75c228a4dff44/results/pure_real_video_qwen_log.json).

### Synthetic-media experiments

`groq_qwen_master_experiment.py` constructs faces with OpenCV drawing operations and generates harmonic audio. Its three named recordings are scenario fixtures, not three independently captured recordings. Its `restore_generative_face` function inserts a sinusoidal intensity change in a mouth rectangle. The function does not run CodeFormer or GFPGAN. `restoration_pipeline.py` explicitly describes its operations as simulations of named restoration families.

The saved `groq_qwen_experiment_log.json` contains 27 cases and 54 processed-candidate comparisons. Recalculation gives 12 correct-to-incorrect transitions, all of which end in **abstention**, not attribution to the wrong person. The logged Qwen responses are therefore evidence of behavior on supplied synthetic measurements, not evidence that a learned face restorer invented speaking motion in real footage.

Recomputed harm-ranking statistics from that log, using the script's negative-score conventions and standard average precision:

| Feature | AUROC | Average precision |
|---|---:|---:|
| Negative confidence | 0.669643 | 0.304483 |
| Negative handcrafted audio/mouth correlation | 0.575397 | 0.381977 |
| Negative input consistency | 0.547619 | 0.300835 |
| Negative weighted input-aware score | 0.476190 | 0.213224 |
| Binary captured/restored disagreement | 0.892857 | 0.571429 |

The README's 0.893/0.571 result matches binary disagreement on these synthetic-media comparisons. It is not an evaluation of the separate `apply_davg_gating` implementation on real learned-restoration failures. Several other README metric values do not match this log. No claim is made that an uninspected historical run could not have produced different values; the current reported numbers need an explicit source before publication.

The root function labelled as a SyncNet baseline computes correlation between audio RMS and pixel differences; it does not load SyncNet. The quality function uses Laplacian/Sobel sharpness rather than MUSIQ. Some printed harm-rejection percentages are literal strings rather than values computed from predictions. These are implementation/reporting issues, not evidence of the named published models' behavior.

Evidence: [synthetic-media runner](https://github.com/bolledduu/video-restoration/blob/08d9ff2a21e88a38da379e1193a75c228a4dff44/groq_qwen_master_experiment.py), [simulation restoration functions](https://github.com/bolledduu/video-restoration/blob/08d9ff2a21e88a38da379e1193a75c228a4dff44/restoration_pipeline.py), [saved log](https://github.com/bolledduu/video-restoration/blob/08d9ff2a21e88a38da379e1193a75c228a4dff44/results/groq_qwen_experiment_log.json).

## 2. The learned-model pilot supplies the relevant empirical motivation

The imported `cpu-pilot/` runner loads official FastDVDnet weights, uses TalkNet, and evaluates BRISQUE and MUSIQ. Its saved audit verifies arithmetic, provenance, label separation, and audio invariance. Those checks do not certify human labels, novelty, or broad generalization.

- A silent track with noise sigma 15 changes from 0/24 errors to 2/24 after NLM; both quality assessors select NLM.
- The same source at sigma 40 changes from 0/24 errors to 5/24 after FastDVDnet; MUSIQ selects it while BRISQUE selects another candidate.
- On another silent track, useful restoration reduces 8/24 errors to 1/24. The fixed disagreement guard rejects that correction.
- Mean frame error is 18.23% for captured, 16.67% for MUSIQ selection, and 19.27% for the fixed guard. The guard does not win the overall comparison.
- Qwen completes eight tool-mediated cases: seven correct majority-speaking answers; selected-video TalkNet frame error equals captured at 18.23%. Qwen does not directly see/hear the clips, and these are not quotation-to-person tasks.
- A speaking source is wrong even in the clean reference. Its restoration regression cannot establish that genuine speaking evidence was destroyed.
- The separate AMI check shows no restoration-induced attribution change. Its transcript-derived activity labels are proxies, not dense independently reviewed ASD labels.
- The predefined two-source harmful-agreement screen remains failed. The larger joint audio-causality claim remains unsupported by that screen.

Evidence: [pilot report](https://github.com/bolledduu/video-restoration/blob/08d9ff2a21e88a38da379e1193a75c228a4dff44/cpu-pilot/QWEN_AGENT_PILOT.md), [measurement runner](https://github.com/bolledduu/video-restoration/blob/08d9ff2a21e88a38da379e1193a75c228a4dff44/cpu-pilot/scripts/agent_pilot_measure.py), [saved measurements](https://github.com/bolledduu/video-restoration/blob/08d9ff2a21e88a38da379e1193a75c228a4dff44/cpu-pilot/results/agent_pilot_measurements.json).

## 3. Architectural implications

1. A useful assessor must distinguish harmful changes from corrections. Unsigned disagreement cannot do that alone.
2. Global audio silence is the wrong trigger for the central failure case: one person can be silent while someone else speaks. The DAVG rule in `breakthrough_methods.py` only accumulates its risk when global audio energy is below 0.05; that condition need not occur in the proposed Alice/Bob example.
3. In the pure-real runner, a gate that accepts only outputs with the same decision as captured is guaranteed to preserve the captured decision. Matching captured accuracy is then a structural consequence, not validation of a useful improvement.
4. Quality, visual fidelity, ASD correctness, and final agent correctness require separate measurements. None substitutes for another.
5. The new contribution should live in assessment of captured speaker evidence. Extra LLMs, a new acronym, hand-selected thresholds, or a weighted combination of existing scores are not enough.
6. Use real restorers and real ASD models for development. Synthetic inserted artifacts may be labelled diagnostic controls, never the sole evidence that the failure arises naturally.

## 4. Scope of this audit

Read-only inspection of current source, reports, protocol/audit files, and saved logs; direct recomputation of root log counts and ranking statistics. No API inference, model retraining, new media annotation, GPU allocation, or paid service use was performed. Existing files and historical results were preserved. The recommended next stage is method development on the measured harm/correction tradeoff, followed by independent evaluation—not another relabeling of the original pilot as a pass.

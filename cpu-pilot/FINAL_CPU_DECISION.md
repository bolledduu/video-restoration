> Historical report. Current completed labeled experiment and research decision: [Final CPU research decision](FINAL_CPU_RESEARCH_DECISION.md).

# Final laptop assessment: NO

**Historical assessment of the original AV-SpeakerBench pilot. The user subsequently reopened the model search; see `HF_CONTINUATION.md` for the newly discovered OmniVR release and supplemental CPU evaluation.**

**Decision: do not invest in a full WACV paper on the basis of this pilot.** The feasibility assessment is closed using the laptop results. No cloud allocation or further pilot preparation is scheduled. This is an investment decision under the available evidence, not an experimental finding that the hypothesized effect is absent.

## Evidence used

Three real clips were processed, six genuine FastDVDnet video-only restoration runs completed, and 18 TalkNet condition runs produced predictions. The final offline analysis verifies paired frame counts, times, face crops, audio hashes, audio-only logits and finite outputs before summarizing them. It uses the previously executed model outputs; no new model inference is claimed.

| Case | PSNR change after FastDVDnet | Visual-only changed frames | Fixed-audio audiovisual changed frames |
|---|---:|---:|---:|
| S1 moderate | -0.798 dB | 5 / 49 | 5 / 49 |
| S1 severe | -0.032 dB | 3 / 49 | 2 / 49 |
| S2 moderate | -0.571 dB | 0 / 47 | 30 / 47 |
| S2 severe | +0.046 dB | 1 / 47 | 5 / 47 |
| S3 moderate | -0.754 dB | 6 / 49 | 0 / 49 |
| S3 severe | +0.146 dB | 5 / 49 | 0 / 49 |

Changed frames compare classifier argmax on degraded versus video-only restored inputs. They are not errors, correct answers or verified speaking states. Original-video predictions are also not ground truth. Audio-only logits are unchanged in all six cases. Cases share three source clips and are not six independent samples.

FastDVDnet improves PSNR in 2/6 cases and SSIM in 3/6. It changes visual-only decisions in 5/6 cases and joint decisions in 4/6, despite using no conditioning audio. Therefore a changed downstream prediction by itself is insufficient evidence for the proposed mechanism. These diagnostics do not establish perceptual improvement, false attribution, audio leakage or a causal explanation of the individual changes.

## Answers to the eight research questions

1. **Clips:** S1 `I0CPeFpSIbE` (280–296 s), S2 `V9WCCw5UWmw` (146–154 s), S3 `gFNigX2jrnM` (1593–1602 s). Exact retrieval, metadata and stream checks were completed. Speaker identities, speaking intervals and published answers remain provisional by user instruction.
2. **Models successfully run:** pretrained FastDVDnet for video-only denoising and pretrained TalkNet for downstream speaker diagnostics, both on CPU. No genuine audio-conditioned restorer ran; the bounded model search found no usable official code/checkpoint for GAVN or DAVD-Net. Hardware is not the established blocker.
3. **Restored videos:** short resized excerpts show mainly smoothing and small texture changes, with mixed fidelity. The denoising model is mismatched to blur/compression and is not a matched audio-conditioning ablation.
4. **Misleading visual speaking cues introduced:** not established. Prediction changes without verified labels do not demonstrate invented speaking motion. The requested audio-conditioned outputs are absent.
5. **Speaker accuracy worse:** not established. Accuracy and false-attribution differences remain null; reporting zero would also be false.
6. **Causal controls survived:** no supporting causal result was established. Fixed downstream audio and shared crops were checked. Matching/counterfactual conditioning, verified natural conflicts and verified identity tracks were not completed.
7. **RQ1/RQ2/RQ3:** RQ1 remains an untested question, not a result. RQ2 can be posed as recoverability but this pilot establishes neither recoverability nor non-identifiability. Do not develop an RQ3 method from these results: no measured gap or advantage over simple rules has been demonstrated.
8. **Full paper:** **NO. Close this pilot and do not allocate additional compute to the current route.** Required evidence for expansion is absent. This rejects further investment on current evidence, not the physical possibility of the proposed failure.

## Reproduce the final assessment

Run `python scripts/finalize_cpu.py` from the repository. It performs no network access and requires only Python's standard library plus the saved JSON outputs. It writes `results/cpu_paired_diagnostics.csv` and `results/final_cpu_decision.json`, including hashes of the input measurements. `RESULTS.md` retains the full execution record; `REPRODUCE.md` covers the original CPU inference.

The original experimental ambition is not fully achieved: genuine audio-conditioned inference and verified accuracy are missing. The laptop feasibility assessment and the requested binary investment decision are complete. There is no pending GPU or annotation request needed to make this decision.

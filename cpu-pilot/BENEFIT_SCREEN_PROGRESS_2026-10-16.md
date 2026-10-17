# Restoration benefit screen: completed and incomplete evidence

**Current recommendation: do not launch the full framework campaign on this evidence. This is not a completed negative Qwen hypothesis test.**

Three fixed two-second source clips (speaking, non-speaking, transition), with noise/FastDVDnet, reduced-resolution/FSRCNN, and JPEG/bilateral-filter pairs. Selection and the investment criterion were saved before inference.

## Complete local detector experiment

All 9 pairs were executed. Captured TalkNet: 176/441 wrong. Restored TalkNet: 179/441 wrong. Corrections: 0; introduced errors: 3.

A label-assisted timestamp selector has 176/441 errors at identical captured coverage: no improvement. This exact finite-candidate upper bound applies to selecting the existing detector answers, not to a Qwen agent's potentially different reasoning or a newly trained predictor/fusion method.

## Actual Qwen evidence

Saved completed sessions: 25/36. Fully paired comparisons: 12/18. Missing sessions: 11. Partial pairs show 0 corrections and 0 introduced errors; they must not be generalized to missing clips or conditions.

Groq returned HTTP 429 for the daily token quota (200,000 tokens/day). The runner is resumable and retains all completed responses. No smaller model, cached detector label, or fabricated prediction was substituted for the requested Qwen output.

The outstanding investment screen depends on the missing agent answers, including the transition clip. It is therefore unscored. Do not call it failed or claim all requested experiments finished.

The complete local bound supplies no evidence of useful detector-level corrections in this candidate set. That supports caution about investing in this particular framework, but does not disprove all restoration benefits, establish novelty, or settle the full Qwen question.

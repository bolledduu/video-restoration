# Capture guard evaluation — 16 October 2026

**Decision: NO to a full-paper campaign for the original audio-created visual hallucination claim or this simple guard as a novel method.**

A narrower restoration-induced tool-agent error is demonstrated. The guard repairs that known example, but does not beat retaining captured evidence. This is a completed retrospective baseline test, not an independent validation study.

## What was tested

Five tracks from three videos, two repetitions each. The inputs are the 20 completed real Qwen responses from the fixed pilot. This script makes no fresh model calls: it implements an automatic post-answer guard. All rules receive only the model answers and detector scores; evaluation labels and reference images are unavailable to the guard.

- Detector-disagreement abstention: return unknown where captured/restored detector decisions differ; abstain on the overall answer if any timestamp differs.
- Detector-disagreement capture: use the captured Qwen tag at each disagreement; keep restored tags elsewhere. Use the captured overall answer if any detector timestamp differs.
- Captured-answer priority: use captured tags whenever answered, otherwise restored tags; use the captured overall answer. This is a conservative baseline, not a novel proposal.

All detector decisions use the existing 0.5 threshold. No training or threshold tuning was performed. Rules were specified after the failures were known, so this cannot be represented as prospective validation.

## Results

| System | Wrong timestamp labels | Answered timestamps /240 | Coverage | Wrong overall answers / answered |
|---|---:|---:|---:|---:|
| captured_only | 0 | 192/240 | 80.0% | 0/10 |
| restored_only | 10 | 96/240 | 40.0% | 2/10 |
| detector_disagreement_abstain | 0 | 86/240 | 35.8% | 0/4 |
| detector_disagreement_capture | 0 | 144/240 | 60.0% | 0/10 |
| captured_answer_priority | 0 | 192/240 | 80.0% | 0/10 |

The 240 timestamp opportunities are 120 unique timestamps evaluated twice, not 240 independent examples. No confidence interval or population success rate is inferred.

## Concrete failure and repair

On `EEey6Pt0rqI:2`, 7.28–8.28 seconds, dataset labels mark the visible person as NOT_SPEAKING. The captured Qwen answer has 0/24 wrong timestamp tags. The restored answer has 5/24 wrong tags (0.00–0.16 seconds relative to the clip). The capture guard replaces those five tags with captured answers and reaches 0/24 errors while answering all 24 timestamps, in both repeats. It does not identify a specific word or prove fabricated mouth motion.

The repair is real on these saved responses. However, captured-only already has zero errors and answers more timestamps across the pilot than the detector-disagreement capture guard. Captured-answer priority matches the captured-only baseline here. Therefore this evaluation establishes no incremental accuracy/coverage benefit over retaining the captured input and no novel contribution.

## Audio-created visual details: mechanism check

FastDVDnet has no audio input; this pipeline cannot test audio-created visual hallucinations.

Audio is used later by TalkNet. That is audiovisual inference, not audio-conditioned restoration.

The restorer receives video frames and a noise level. Its callable interface and actual generation calls were inspected and recorded with source hashes. Changing audio is not an input intervention to this restorer. These results neither demonstrate nor disprove the hypothesis for a genuinely audio-conditioned restoration model.

## Research decision

- Narrow failure motivation: supported by the specific timestamp counterexample.
- Repair of the known failure: supported for the deterministic capture guard.
- Better method than captured-only: not demonstrated; the tested guard is worse in coverage at the same zero observed error count.
- Audio-created visual hallucinations: not tested by this architecture; reject this claim for the current results.
- Full-paper investment on the current method/claim: NO. Do not scale this baseline merely to obtain a positive result.

The scientifically defensible next contribution would need to preserve a genuine benefit of restoration while avoiding these errors, beyond simply keeping captured evidence. That is a future falsifiable requirement, not a result of this pilot. The earlier failed majority-answer replication criterion remains failed and is not replaced by this post hoc guard result.

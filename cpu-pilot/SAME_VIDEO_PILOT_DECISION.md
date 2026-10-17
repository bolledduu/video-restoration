# Same-video pilot: full-paper commitment is not supported

**Decision: NO to committing to the proposed full-paper experiment campaign on the current evidence.** The fixed pilot criterion fails. A reproducible failure exists, but it does not repeat across enough sources to pass that criterion. This is an investment decision for the tested setup, not a proof that restoration cannot harm speaker detection.

## Completed evaluation

Five labelled UniTalk source videos, seven tracks, eleven fixed degradation conditions, and 264 aligned frame decisions were evaluated. Four conditions are speaking and seven are silent. Every selector has full coverage. No videos, noise settings, or thresholds were changed to obtain failures.

The preceding run recomputed the restorations and TalkNet predictions. This completion reuses those audited outputs, checks their file hashes, computes the nine missing BRISQUE/MUSIQ frame measurements for the three additional FastDVDnet outputs, and evaluates selection and risk detection. It is a same-video analysis, not a new independent replication. There were no new agent API calls.

The original eight-case test keeps its original four-candidate pool and its original pass criterion. The expanded eleven-case comparison uses the three candidates available for every case: captured input, spatial NLM, and FastDVDnet with the true noise sigma. The fixed-sigma-55 variant remains in the historical test but is not silently mixed into the common-pool comparison.

The three AV-SpeakerBench videos from the teammate's branch remain prediction-change diagnostics because their labels are provisional. The DMD gaze records do not test speaker attribution.

## RQ1: harmful quality-assessor agreement

The original criterion requires harmful BRISQUE/MUSIQ agreement on **at least two independent sources**, plus a restoration benefit on at least one source.

| Requirement | Observed | Outcome |
|---|---:|---|
| Harmful agreement on at least two sources | One: `EEey6Pt0rqI` | Fail |
| Some restoration benefit on at least one source | One: `kgVz_AXYvn0` | Pass |
| Combined original criterion | Both requirements needed | **Fail** |

The expanded common-candidate comparison also finds harmful agreement on only `EEey6Pt0rqI`. The three additional videos have no new errors. The known silent-track error remains real under the dataset labels, but is not a broad replicated effect.

## RQ2: identifying harmful candidates

The target is whether a restored candidate introduces at least one correct-to-incorrect frame relative to captured input. Four of the 22 noninput candidate conditions satisfy this definition. These candidate conditions share sources and inputs; they are not 22 independent samples.

| Fixed baseline risk score | AUROC |
|---|---:|
| Captured/restored prediction disagreement | 0.924 |
| Low model confidence | 0.722 |
| Pixel MSE from captured input | 0.694 |
| Low MUSIQ | 0.653 |
| High BRISQUE | 0.361 |

These are descriptive, untrained baseline rankings on reused data. They are not held-out performance for a novel assessor. On the seven conditions where TalkNet is correct on every source-reference frame, disagreement and uncertainty both rank the two harmful candidates perfectly. That very small result is evidence that simple baselines are strong here, not evidence of perfect generalization.

Thus, there is a measurable risk signal, but **no demonstrated improvement beyond existing disagreement/confidence baselines**. No new assessor was trained or validated in this pilot. Lip-synchronization and audio-conditioning mechanisms were not newly tested by these measurements.

## RQ3: useful selection at full coverage

Errors, new errors, and corrections below count scored frames. Higher MUSIQ is better. It is a visual-quality score, not a human preference judgment. The error denominator is 264 for every row.

| Selection rule | Errors | New errors | Corrections | Mean MUSIQ |
|---|---:|---:|---:|---:|
| Keep captured input | 35 | 0 | 0 | 15.331 |
| Always spatial NLM | 51 | 26 | 10 | 16.240 |
| Always FastDVDnet | 56 | 29 | 8 | 17.933 |
| Highest MUSIQ | 34 | 7 | 8 | 18.217 |
| Best BRISQUE | 58 | 26 | 3 | 16.013 |
| Highest model confidence | 49 | 24 | 10 | 15.997 |
| Require quality-assessor agreement | 37 | 2 | 0 | 15.694 |
| MUSIQ with original 10% disagreement guard | 37 | 2 | 0 | 17.758 |
| MUSIQ with zero-disagreement guard | 35 | 0 | 0 | 17.395 |
| Label-informed oracle, evaluation only | 25 | 0 | 10 | 17.931 |

The original guard performs worse than keeping the input. Quality-only selection slightly reduces total error while introducing seven new errors; total accuracy alone hides those regressions.

The strict guard retains captured input whenever the quality winner changes any binary frame decision. It therefore preserves the detector's existing predictions by construction. It raises mean MUSIQ without new errors, but cannot correct any captured-input decision. **This is a simple baseline, not a novel safeguard.** The oracle's ten possible corrections show room in this candidate pool, not an achieved method result.

BRISQUE, SSIM, source-balanced error, and per-source error are included in the machine-readable results. Five sources with uneven condition counts do not justify population confidence or frame-level significance claims.

## Detector-limitation sensitivity check

The source-reference-correct subset contains seven cases, four sources, and 168 scored frames. Captured input has zero errors in this subset. MUSIQ selection introduces seven errors; FastDVDnet introduces five and NLM two. The 10% guard leaves two errors; the strict guard leaves zero.

This confirms that the known harmful examples are not solely caused by the detector already failing on the source reference. However, every harmful example in this subset still comes from the same source. Since captured input is already correct throughout this subset, it cannot demonstrate task-accuracy gains from restoration.

## Decision and scope

The pilot supplies a concrete failure example and a useful baseline comparison. It does **not** pass the original replication criterion, establish a new method advantage, or establish audio-caused false corroboration. FastDVDnet receives no conditioning audio, and TalkNet frame errors are not Qwen quotation-attribution errors.

The requested decision is therefore **NO to full-paper commitment from this pilot**. The initial pilot can be closed with this result. Repeatedly rerunning these same videos or increasing their degradation until the outcome changes would not supply independent confirmation.

Earlier “GO” wording referred to permission to explore a possible method. It should not have been read as a passed hypothesis test. This report makes the investment recommendation explicit and supersedes that broad wording; it does not change any prior measurement or the research questions.

## Reproducibility

- [Fixed completion protocol](same_video_complete_protocol.json)
- [Complete results and per-case records](results/same_video_complete_pilot.json)
- [Selection table](results/same_video_complete_pilot.csv)
- [Evaluation script](scripts/complete_same_video_pilot.py)
- [Prior same-data branch reproduction](TEAMMATE_BRANCH_RECHECK_2026-10-15.md)

The script validates candidate-array and quality-checkpoint hashes, reproduces the original gate from its saved measurements, checks paired error arithmetic, and verifies that the strict guard preserves captured-input decisions. Missing quality scores are evaluated using the same checkpoints and frame positions as the prior runs. Cached restorer/detector outputs remain explicitly identified as reused.

# Decision and fixed next experiment

**YES: proceed with an independent-source study of quality–attribution failures. NO: do not present the current lower-face score or patch replacement as a validated new framework.**

The completed preparation changes what we should build. We have reproducible restoration-induced errors and a measurable spatial sensitivity. We do not have evidence that a simple mouth-motion risk score is useful beyond prediction disagreement. The larger study should test the failure and its explanation, not assume a successful method.

## Work completed now

- Re-ran captured and restored TalkNet inference and verified agreement with saved scores within 1e-6 on all 17 comparisons from seven tracks/five sources.
- Ran two complementary spatial interventions for each comparison: 68 inferences total. Audio was identical in every condition, and audio-only predictions were invariant.
- Kept all cases: 236 captured errors versus 269 restored errors over 633 repeated timestamp opportunities; 33 new errors, zero corrections.
- Lower-half replacement rescued 20 introduced errors, but had 253 total errors: worse than captured-only's 236. Upper-half replacement rescued 13, but had 274 total errors. Rescue counts alone would hide new hybrid errors.
- On the qv3 speaking track, lower replacement rescues 18/18 FastDVDnet-introduced errors; upper replacement rescues 0/18. For spatial NLM on that track, lower rescues 1/12 and upper rescues 12/12. This is restorer-dependent sensitivity, not a universal mouth mechanism.
- The frozen exploratory requirement of lower-half superiority on two sources failed: only one source met it. Do not reinterpret the same test as a successful universal mechanism.
- Fixed lower-region temporal residual AUROC was 0.538, versus 1.000 for prediction disagreement on these 17 exploratory candidates. There were no restoration corrections, so disagreement has an unusually easy target here. These are descriptive values, not held-out performance or a claim that disagreement always works.
- Audited 69 saved Qwen sessions, separating the three cohorts. In the complete 24-session transfer cohort, Qwen repeated 48 FastDVDnet tool errors, abstained on 48, and corrected none. Under spatial NLM it corrected two tool-error occurrences, but this is not evidence that restoration corrected a captured-agent error. See the separate propagation audit.
- Prepared a fixed list of 12 sources absent from the nine-source local pilot cache, using a published-dataset revision and deterministic hash ordering. Downloaded annotations and selected 32 four-second windows without model predictions; four sources lack an eligible transition category under the frozen rules. Media eligibility still requires verification; no inference on these new sources has run. Before media exclusions, this is 64 degraded inputs, 128 restored outputs, and 384 paired-condition Qwen sessions with two repeats.

The 25/36-session benefit test remains incomplete and is not counted as a completed negative result. No API model calls were made in this follow-up.

## Four questions for the independent study

1. Under which tested restoration conditions do increases in visual-quality scores accompany new speaker-attribution errors?
2. How does the spatial location of restoration changes affect these errors, and does the effect differ by restorer?
3. When supplied these perception-tool outputs, does Qwen propagate errors, correct them, introduce additional errors, or abstain?
4. Do captured/restored temporal-change measures add useful risk information beyond global quality, detector uncertainty, and prediction disagreement on held-out sources?

These are testable questions, not four positive claims. Anatomical mouth localization and temporal causality are not established by replacing horizontal crop halves. A named architecture is premature. A simple combination of existing cues is not a verified novelty contribution.

## Fixed execution order

1. **Media preflight.** Use `failure_confirmation_protocol.json` and `data/failure_confirmation/manifest.json`. Fetch the pinned archives for the 12 selected sources. Verify native crop size, complete frames, track audio offsets, duration and annotation agreement before inference. Reject ineligible windows with an explicit reason. Do not substitute sources after seeing outcomes. Video archives total approximately 1.62 GB; audio is additional.
2. **Independent local matrix.** Run the eligible four-second speaking, non-speaking and transition tracks at Gaussian sigma 10 and 25. Compare captured inputs, FastDVDnet with the matching noise parameter, and fixed spatial NLM. This is one degradation family and one learned plus one classical restorer; do not claim broad family generalization. Score unmodified source input as a diagnostic. Reuse each source soundtrack exactly.
3. **Quality and attribution table.** Report source/track counts, MUSIQ and BRISQUE deltas, captured/restored error rates, introduced-error rate, correction rate, and the joint quality-improvement/error event. Include all conditions, not only harmful examples. Give source-level results and source-cluster uncertainty. Report misses and false-speaking events separately.
4. **Mechanism table.** Repeat the same upper/lower interventions without tuning masks. Report both rescues and all-case errors, perturbation magnitude and score movement. Add landmark masks only in a separately specified follow-up; do not silently replace the current proxy after seeing results.
5. **Agent table.** Once local tool payloads are frozen, run the paired Qwen conditions with the same evidence access and two repeats. Include failed requests, answer coverage, propagated errors, genuine corrections and newly introduced errors. The maximum matrix is 432 agent sessions, before any extra recovery calls. It is not a quick free-API pilot, and should not be launched against an exhausted daily quota. A second agent is required before claiming agent-model generalization; none is silently substituted here.
6. **Assess risk scores once.** Evaluate the already fixed scores on the independent sources. A new fitted estimator needs separate development sources, frozen hyperparameters and untouched test sources. Include disagreement and uncertainty, not only weaker IQA baselines. If no incremental signal appears, report that result rather than building a framework around it.

The larger batch is an actual confirmation study, not another round of selecting a few failure videos. Its sample size is an initial bounded scope, not a power calculation or a guarantee of conference acceptance. Stop after the fixed batch to assess effect sizes and uncertainty; do not keep expanding until significance or positive results appear.

## What would justify a method paper?

Independent replication would support the problem. A method contribution additionally needs useful information unavailable from simple disagreement/confidence, and improvements on an untouched source split at matched coverage. Selective restoration must beat captured-only on the outcomes it claims to improve. Neither an oracle nor patch rescue alone satisfies that requirement.

For now, the supported direction is a careful empirical quality-assessment study. Publication novelty remains to be established against restoration-fidelity and ASD robustness literature. LASER already uses lip landmarks for robustness; lip attention itself is not new: https://openaccess.thecvf.com/content/WACV2026/html/Nguyen_LASER_Lip_Landmark_Assisted_Speaker_Detection_for_Robustness_WACV_2026_paper.html

## Evidence files

- `LOCALITY_SCREEN_DECISION_2026-10-16.md`: all intervention results and limitations.
- `results/locality_measurements.json`: actual scores, labels, hashes and interventions.
- `results/locality_audit.json`: arithmetic and descriptive risk-score audit.
- `results/locality_intervention.png`: source-level illustration.
- `AGENT_PROPAGATION_AUDIT_2026-10-16.md`: real Qwen response reanalysis.
- `failure_confirmation_protocol.json`: fixed independent-source study configuration.
- `data/failure_confirmation/manifest.json`: annotation-selected queue, explicitly not completed experiments.

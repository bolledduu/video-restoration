# Prompt 2: Rigorous Audiovisual Restoration Safeguard Implementation & Evaluation

Implement and evaluate a rigorous audiovisual restoration safeguard in this repository. Work as a research engineer accountable for every reported claim. Do not optimize for positive-looking results.

Read the repository instructions, current research questions, METHOD_PROPOSAL_2026-10-15.md, reviews/FIXES_2026-10-17.md, reviews/REVIEW_2026-10-17.md, and results/EXPERIMENT_VALIDITY.json under cpu-pilot. Verify that the repairs from commit 2726ae0 are present.

Use the downloaded datasets under:
`C:\Users\bolle\Final Experiments\video-restoration\cpu-pilot\data`

Preserve existing files and results. Inspect local changes before editing.

---

## OBJECTIVE

Determine whether a captured-evidence verifier can identify restoration-induced speaker-attribution errors and preserve useful restoration benefits in a tool-using multimodal-agent pipeline.

Do not claim that appearance quality, captured-input consistency, or agreement between models establishes correctness. Do not call the framework novel or validated before the supporting comparisons exist.

---

## FRAMEWORK

Implement these stages as separate, testable components:

1. Captured evidence and candidate generation.
2. Captured-support measurement.
3. Attribution-influence measurement.
4. Learned harm/correction assessment with independent calibration.
5. Candidate selection and optional abstention.
6. Agent integration and paired evaluation.

The main scientific hypothesis is that captured support and attribution influence together contain useful information about restoration-induced errors beyond existing quality, confidence, disagreement and input-consistency features.

---

### A. CAPTURED EVIDENCE

Preserve the original audio, timestamps, face identities and captured frames. Align every candidate to the same person and scoring grid. Use identical audio across primary captured/restored comparisons.

Distinguish original-source video, synthetically degraded video and restored video in every artifact. A compressed source is a comparator, not automatically pristine ground truth.

Validate face tracks, audiovisual alignment, annotation applicability and speaker-state labels before scoring. Record exclusions and their reasons. Never silently repair labels based on predictions.

---

### B. CAPTURED-SUPPORT VERIFIER

Implement the masking experiment from the method proposal, but treat its output as a diagnostic—not a correctness certificate.

Choose masks, regions, seeds, timing controls and aggregation rules before inspecting outcomes. All channels at a withheld pixel site must be hidden together. Withheld values must not leak through filling, normalization, alignment, cached features or restoration inputs.

Generate audit restorations using only retained captured measurements. Evaluate their predictions on the withheld measurements. Compare against an appropriate baseline built from the same retained information.

Measure whether the temporal restoration increment is better supported at its original timing than under prespecified timing-disruption controls. Compare all controls on identical valid timestamps; avoid circular wraparound and unequal boundary losses.

Include controls for:
- Static appearance changes.
- Head motion.
- Non-mouth regions.
- Simple held-out prediction error.
- Ordinary captured/restored consistency.
- Masking sensitivity.

Do not assume masking remains statistically independent after compression, resizing or correlated noise. State which assumptions hold in each experiment. Do not advertise an unbiased estimator or causal guarantee without justification.

Critically, measure whether the masked audit restoration represents the actual unmasked candidate. If masking substantially changes its behavior, flag the diagnostic as unreliable rather than transferring its score unquestioningly.

---

### C. ATTRIBUTION INFLUENCE

Measure how the actual candidate changes the downstream speaking scores and decisions with audio held fixed.

Keep these quantities separate:
- Appearance improvement.
- Captured-evidence support.
- Attribution-score change.
- Threshold-decision change.
- Ground-truth harm or correction.

Attribution influence is not correctness. Weak support is not proof of hallucination. Prediction disagreement is not proof of error.

Do not claim audio-created visual hallucinations from a video-only restorer. Any audio-intervention diagnostic must be executed explicitly and logged; never replace it with constants or probability-difference proxies.

---

### D. RISK ESTIMATION AND SELECTION

Use support and influence measurements to estimate the four transitions:
- captured correct / restored correct;
- captured correct / restored incorrect;
- captured incorrect / restored correct;
- captured incorrect / restored incorrect.

Begin with transparent, low-capacity models and strong simple baselines. Add complexity only when development evidence supports it.

Use separate source-level training, calibration and test partitions. Keep all clips, people, degradations and restoration alternatives from the same source within its assigned partition.

Previously inspected pilot sources are development evidence. Fit preprocessing on training only. Choose thresholds on calibration only. Do not reuse calibration outcomes for unrestricted model development.

Report class support before fitting. If harms or corrections are absent or too sparse, do not claim that the corresponding predictor is validated.

Select among captured input and eligible restorations using calibrated attribution risk and a prespecified quality preference. Always include captured input. Evaluate abstention separately so it cannot disguise poor selection.

Do not infer selected-video quality from class agreement or a weighted average of whole-clip IQA scores. For mixed temporal selections, construct and evaluate the actual output, including temporal artifacts, or report quality as unavailable.

---

### E. AGENT INTEGRATION

Expose the verifier through a typed tool that returns measured diagnostics, uncertainty/status and provenance—not labels or evaluation errors.

Precisely describe the agent’s access: tool measurements, images, audio or some combination. Do not describe a text-only tool agent as directly understanding video.

Compare the same model and controller under:
1. Existing tools without the verifier.
2. Existing tools plus ordinary consistency diagnostics.
3. Existing tools plus the proposed verifier.

Freeze prompts, candidate availability, decoding settings and tool budgets. Include a budget-matched comparison. Log actual usage and incomplete episodes.

Test whether the agent uses the verifier appropriately. A correct tool result does not establish a correct agent answer.

---

### F. EXPERIMENTAL COMPARISONS

Include captured-only, each fixed restorer, MUSIQ selection, BRISQUE selection, quality consensus, confidence selection, disagreement fallback, disagreement abstention, simple input-consistency selection, and a trained baseline using the same data without the proposed support features.

Include an oracle solely as a nondeployable upper bound.

Compare support-only, influence-only and combined variants. Report what each component adds beyond the strongest simple baseline, not merely beyond captured-only.

Use a frozen sampling rule across speaking, listening and transition cases. Retain no-harm cases. Never select examples because they favor our method.

---

### G. METRICS AND INFERENCE

Report attribution errors, newly introduced errors, corrections, coverage and captured performance on exactly the same answered timestamps.

Report restoration quality separately from attribution correctness. IQA improvements are metric improvements unless independently validated by human judgments.

Report per-source results and paired source-clustered uncertainty. Repeated frames, candidate restorations and repeated model runs are not independent source samples.

For risk prediction, report discrimination and calibration with event counts. Mark metrics undefined when their required classes are absent.

Keep detector outcomes separate from agent outcomes. Preserve incomplete requests and API failures as failures, not zero-error observations.

---

### H. EXECUTION ORDER

First implement one complete development-only path:
real clip → restoration → support measurements → influence measurements → selector → agent answer → evaluator.

Run tests for leakage, identity alignment, withheld-value invariance, scoring-grid completeness, evidence origin, calibration separation and resume consistency.

Then run a small frozen development batch containing both speaking and listening examples. Check whether the proposed diagnostic adds useful information beyond the controls.

Only after integration and diagnostic checks pass, freeze the evaluation configuration and run the full experiments. Start with one exact model across datasets, then expand models using the unchanged protocol.

If the diagnostic does not work, investigate its mechanism and assumptions on development data. Do not hide this with a larger model, easier videos, different thresholds chosen on test data, or renamed claims.

---

## DELIVERABLES

Provide executable code, tests, immutable configurations, data/checkpoint hashes, raw predictions, tool traces, exclusion records and automatically generated tables.

Maintain a claim-to-evidence table distinguishing:
- Implemented behavior.
- Development observations.
- Held-out findings.
- Untested hypotheses.

Before claiming novelty, inspect current primary-source literature and identify the exact technical distinction and necessary baseline. Combining established components may be legitimate research, but the contribution must be accurately attributed and experimentally demonstrated.

Do not stop at a plan when implementation can proceed. Do not describe this as a successful final framework until the measurements support that conclusion.

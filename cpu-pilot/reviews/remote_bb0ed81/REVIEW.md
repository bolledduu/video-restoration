# Review of pushed results at bb0ed81

Reviewed the exact origin/main snapshot in an isolated directory. Local development code and completed CPU runs were preserved. No remote experiment files were modified.

## Decision

The pushed package contains real saved agent answers with reproducible arithmetic, but it does not validate the claimed masked CTS framework. Correct the measurement provenance and evaluation before spending on more full runs or using these tables as final paper results. These implementation defects are not a scientific rejection of the research questions.

## Critical findings

### 1. The executed CTS experiment uses constructed proxies

scripts/run_cts_pilot.py:190 computes support as 2*delta_musiq - max(0,delta_brisque) - 10*mean_probability_change. Lines 198-200 construct held-out MSE and consistency from score changes and set mask_stability=15.0. The actual masked restoration function is imported but not called in the production loop. The unit leakage check uses random arrays and a roll function.

I independently checked the saved measurements: all 44 restored-candidate CTS scores match this formula, and all 44 mask-stability values are exactly 15.0. They are not measured masked-audit diagnostics. Reclassify these runs as proxy-tool development experiments, preserve them, and regenerate CTS values from real masked restoration calls.

### 2. OmniVR labels and quality metrics do not support the claims

scripts/add_omnivr_to_measurements.py:39 thresholds the reference detector's predictions to create labels. These measure agreement with a reference detector, not independently established speaker truth. Lines 45 and 67 store PSNR under musiq; BRISQUE fields contain 100*(1-SSIM). The importer also constructs CTS scores from a formula.

Keep actual PSNR and SSIM under their correct names. Either obtain independent speaking annotations and actual CTS measurements, or limit this cohort to explicitly labelled reference-prediction consistency diagnostics. Do not claim independently correct or wrong speaker attribution from these pseudo-labels.

### 3. Agent tool bypasses the trained risk assessor

scripts/ava_evidence_environment.py:101 recommends restoration when support >= 0 and validity == VALID. It does not apply the fitted harm/correction assessor or calibrated threshold. It returns provenance withheld_pixel_temporal_audit even when its inputs were constructed proxies. Missing diagnostics default to zero and VALID, potentially returning a positive recommendation without a measurement.

Require measured, complete provenance; fail closed on missing diagnostics. Use the frozen trained/calibrated artifact for recommendations, and test that the agent receives that exact artifact's result. Do not describe this threshold heuristic as the trained framework.

### 4. The supposed held-out policy table pools all partitions

scripts/run_cts_pilot.py:236 evaluates all processed cases. The table has 705 timestamps: 441 training, 120 calibration and only 144 nominal test timestamps. Source separation during fitting does not make a pooled table held-out. All sources were already inspected in historical pilots, and comments assign sources using their known harms/corrections.

Report development partitions separately and use fresh, frozen source groups for confirmation. Never rank methods using a training/calibration/test pooled score labelled held-out.

### 5. Claimed superiority contradicts the policy table

selection_summary.json gives CTS 216/705 errors versus captured 207/705, MUSIQ selection 213/705 and the named consistency baseline 207/705. The report's statement that CTS has lower error than captured is false. It does improve on always-restored at 230/705, but that is a weaker comparison and still uses constructed features.

The calibration code chooses the 80th percentile of predicted risks without using calibration harm outcomes; target_max_harm is unused. It does not establish a 15% harm constraint or calibrated event probabilities. The named CDI baseline is a score-change heuristic, not an implementation of a cited published fidelity method.

### 6. Lower OmniVR error is explained by coverage

CTS answers only 98/147 timestamps and abstains on the entire 49-timestamp speech_001 clip containing all 11 captured errors. Captured has 0/98 errors on the same answered timestamps. The ordinary-consistency condition answers all 147 timestamps with 0 errors under these pseudo-labels.

The combined table compares CTS 17/194 against baseline 39/243 and consistency 28/243. These are different coverage levels. The table must show eligible and answered denominators and compare the same answered timestamps or matched coverage. Do not describe the conditional error-rate reduction as full-coverage superiority.

### 7. Current selection cannot establish generalization

The four UniTalk cases are explicitly chosen in code using known harm/correction outcomes. AVA and AMI were removed from main tables after constant results were observed. Keep these as development cases; show all prespecified datasets and justify any benchmark inclusion independently of success. OmniVR sample stems also do not establish independent original source-video groups.

## Independently rescored agent outcomes

| Cohort | Baseline errors / answered | Consistency errors / answered | CTS-proxy tool errors / answered | Captured comparator |
|---|---:|---:|---:|---:|
| AVA | 107/297 | 107/297 | 107/297 | 118/297 |
| AMI | 24/48 | 24/48 | 24/48 | 24/48 |
| UniTalk | 28/96 | 28/96 | 17/96 | 14/96 |
| OmniVR, detector pseudo-labels | 11/147 | 0/147 | 0/98 | 11/147 overall; 0/98 on CTS answers |

All saved episode error counts matched recomputation. The UniTalk proxy-tool condition reduces errors relative to the agent baseline but remains worse than captured, introduces 3 new errors and retains no corrections. It is an exploratory agent response to added proxy information, not demonstrated masked-verifier benefit. These agent runs are tool-measurement-only, not direct Qwen video perception.

## What to do next, in order

1. Preserve and mark the affected CTS/OmniVR tables as exploratory proxies; fix the unsupported report conclusions before paper use.
2. Connect real masked audit execution to production measurements. Require source/input/weight hashes, masks, held-out residuals, temporal-control predictions and full-versus-mask stability. Remove all proxy substitutions and fabricated provenance.
3. Separate genuine labels from detector pseudo-labels, and use correctly named/computed quality metrics. Reuse valid raw media and predictions instead of repeating downloads.
4. Fit the same assessor used by the agent. Validate harm/correction support on training sources, select thresholds using independent calibration outcomes, and freeze the artifact.
5. Run fresh source-disjoint comparisons with captured-only, quality selection, confidence/disagreement, a correctly identified consistency baseline, and matched-feature ablations. Report all planned datasets and source-level uncertainty. Keep direct audiovisual VLM tests separate from tool-agent tests.
6. Generate tables from validated test predictions with both eligibility and answer coverage visible. Paper success claims depend on these new results; unit-test success is insufficient.

The separate local actual masked-audit run has completed 11 development cases, and the local new AVA training run has completed three clips. These were not part of bb0ed81 and are not counted as evidence supporting the pushed tables. They provide a starting point for replacing the proxy measurements.

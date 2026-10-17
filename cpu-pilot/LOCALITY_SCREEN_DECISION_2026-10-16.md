# Local restoration intervention: completed exploratory screen

**Decision: Do not build a lower-face-specific framework on this screen; broader quality-attribution failure study remains separately motivated.**

All 17 cached restoration comparisons were retained: seven tracks from five source videos. This reuses known pilot material and is exploratory. It does not add independent source replication or new Qwen responses.

We reproduced captured/restored TalkNet scores, then replaced a fixed lower or upper half of the restored crop with captured pixels. Masks have equal area and identical feathering. Audio is identical. Regions are coarse face-crop halves, not verified anatomical mouth masks.

| Source | Comparisons | Captured errors | Restored errors | New errors | Corrections | New errors rescued: lower | New errors rescued: upper |
|---|---:|---:|---:|---:|---:|---:|---:|
| ABqo634gI5s | 4 | 48/96 | 48/96 | 0 | 0 | 0 | 0 |
| Pc5iQdNpDX0 | 3 | 147/147 | 147/147 | 0 | 0 | 0 | 0 |
| QRXbRSq8p4s | 3 | 0/147 | 0/147 | 0 | 0 | 0 | 0 |
| ho_oK12hY6M | 3 | 29/147 | 32/147 | 3 | 0 | 1 | 1 |
| qv3-HaaxGUc | 4 | 12/96 | 42/96 | 30 | 0 | 19 | 12 |

Totals: 33 introduced-error occurrences, 0 corrections. Lower replacement rescues 20 of these errors; upper replacement rescues 13. These rescues can overlap. All-case hybrid errors are 253 (lower retained) and 274 (upper retained), compared with 236 captured and 269 restored.

Frozen exploratory gate: lower replacement rescues more harmful timestamps than upper replacement on at least two sources. Result: FAIL; supporting sources: ['qv3-HaaxGUc'].

Equal-area patches do not guarantee equal perturbation energy, and compositing can introduce artifacts. Spatial intervention does not isolate temporal causality. Source position, pose, occlusion, model context and original detector errors remain alternative explanations. This is a sensitivity experiment, not proof of invented mouth motion.

## Fixed risk scores (descriptive only)

Target: any captured-correct/restored-wrong timestamp in a candidate. Higher risk scores were defined before intervention inference. AUROC here uses all 17 cached comparisons and is not a generalization estimate. No direction is flipped after seeing results.

| Score | Candidate-harm AUROC |
|---|---:|
| lower_spatial_change | 0.423 |
| lower_temporal_change | 0.538 |
| upper_spatial_change | 0.423 |
| upper_temporal_change | 0.442 |
| global_spatial_change | 0.442 |
| global_temporal_change | 0.462 |
| musiq_improvement | 0.385 |
| brisque_improvement | 0.635 |
| binary_disagreement | 1.000 |
| restored_uncertainty | 0.750 |

No learned assessor, held-out incremental advantage, useful agent safeguard, or paper novelty has been established. Frame counts are repeated opportunities, not independent samples. Full experiments must split by source and report source-level uncertainty.

## Existing work to distinguish

LASER already incorporates lip landmarks for speaker-detection robustness. Our proposed question concerns restoration-induced errors and the adequacy of quality assessment; merely attending to lips is not a novelty claim. https://openaccess.thecvf.com/content/WACV2026/html/Nguyen_LASER_Lip_Landmark_Assisted_Speaker_Detection_for_Robustness_WACV_2026_paper.html

TalkNet is the perception tool used here; Qwen is a separate agent receiving tool measurements. https://github.com/TaoRuijie/TalkNet-ASD

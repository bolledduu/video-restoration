# Solution mechanism experiment

Completed all 17 cached comparisons from five sources: 153 detector calls and zero new Qwen calls. Prototype gate passes: [].

| Video supplied to detector | Errors /633 | New errors vs captured | Corrections vs captured | Mean MUSIQ change | Mean BRISQUE change | Both quality scores improve: pairs /17 |
|---|---:|---:|---:|---:|---:|---:|
| captured | 236 | 0 | 0 | +0.000 | +0.000 | 0 |
| restored | 269 | 33 | 0 | +4.003 | -9.305 | 12 |
| static_residual | 236 | 2 | 2 | +0.019 | -2.343 | 5 |
| dynamic_residual | 272 | 37 | 1 | +2.411 | -30.607 | 12 |
| half_blend | 257 | 21 | 0 | +0.689 | -14.101 | 10 |

MUSIQ higher and BRISQUE lower are better. Mean deltas average pairs, not independent sources. All variants answer every timestamp; no improvement can arise from abstention. The 633 opportunities repeat source timestamps across restoration conditions.

Static residual: captured video plus the time-average restoration residual. Before clipping and rounding this preserves captured pixel differences across frames. Dynamic residual: captured video plus only the time-varying residual; this preserves restored pixel differences before clipping. Neither is a semantic appearance/motion decomposition; head motion, noise and clipping are confounds. Half blend is an ordinary control.

Frozen solution screen: fewer total errors than captured-only, no source worse than captured-only, positive mean MUSIQ change and negative mean BRISQUE change. This is an exploratory engineering gate, not significance, novelty certification, or validation on untouched sources.

## Audio-timing response test

For both captured and restored video, rerun the detector with original audio and circular shifts of +/-0.32 seconds. Measure how restoration changes the original-minus-shifted response contrast. Audio-only scores match exactly between captured and restored video for each audio condition. A change in contrast is model sensitivity, not proof of genuine synchronization or invented speech.

| Risk score | AUROC for any introduced error |
|---|---:|
| alignment_response_change | 0.750 |
| binary_disagreement | 1.000 |
| absolute_score_change | 0.692 |

These AUROCs use the same 17 exploratory pairs. With no beneficial original restorations, disagreement can perfectly rank harmful candidates without demonstrating a useful general solution. The timing test must add value on independent sources containing both improvements and regressions before it can justify a new assessor.

## Per-source errors

| Source | Captured | Restored | Static residual | Dynamic residual | Half blend |
|---|---:|---:|---:|---:|---:|
| ABqo634gI5s | 48 | 48 | 48 | 47 | 48 |
| Pc5iQdNpDX0 | 147 | 147 | 147 | 147 | 147 |
| QRXbRSq8p4s | 0 | 0 | 0 | 0 | 0 |
| ho_oK12hY6M | 29 | 32 | 30 | 34 | 31 |
| qv3-HaaxGUc | 12 | 42 | 11 | 44 | 31 |

The previously prepared 12-source confirmation queue was not used to design or select these prototypes. No novel framework is claimed from this screen.

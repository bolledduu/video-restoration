# Corrected AVA CPU study: validated results

This is a deterministic FastDVDnet/TalkNet-AVA study on a source-disjoint partition of **AVA validation** recordings with Gaussian noise at sigma 25. It is not an AVA hidden-test result, an LLM-agent evaluation, a human-quality judgment, or a transfer study.

The frozen selector was created before test inference (SHA-256 `5e25f3f19d787934fa6e63a02d923bb3fde270024cb44d173f96f393dfe2cc11`). The independent verifier checked that training, calibration, and test source IDs are disjoint; that the frozen model and calibration hashes match; that the selector was frozen before the first test event; and that every saved policy count recomputes from the test predictions and labels. The eight saved code hashes match the final code snapshots. Four of 42 planned test clips had recorded technical exclusions, leaving **38 clips, 15 sources, and 3,762 eligible timestamps**. Every policy below makes a detector decision at every eligible timestamp.

| Policy | Errors / 3,762 | New errors | Corrections | Restored clips / 38 | Mean delta MUSIQ | Accuracy gain vs. captured, 95% source-bootstrap CI (percentage points) |
|---|---:|---:|---:|---:|---:|---:|
| Captured | 340 | 0 | 0 | 0 | 0.000 | [0.00, 0.00] |
| FastDVDnet restored | 257 | 103 | 186 | 38 | 9.116 | [-1.94, 7.53] |
| MUSIQ selection | 284 | 82 | 138 | 36 | 9.191 | [-2.53, 7.34] |
| BRISQUE selection | 383 | 92 | 49 | 26 | 7.591 | [-2.75, 0.23] |
| Quality-consensus selection | 410 | 71 | 1 | 24 | 7.667 | [-3.22, -0.65] |
| Confidence selection | 322 | 48 | 66 | 17 | 3.132 | [-1.32, 2.28] |
| Disagreement fallback | 340 | 0 | 0 | 0 | 0.000 | [0.00, 0.00] |
| Base assessor selector | 373 | 35 | 2 | 21 | 4.019 | [-2.31, 0.03] |
| Ordinary-audit selector | 376 | 38 | 2 | 19 | 3.349 | [-2.27, -0.05] |
| Timing-support selector | 345 | 5 | 0 | 9 | 3.302 | [-0.35, 0.00] |

The restored-only branch has fewer pooled errors than captured, but it also introduces 103 errors while correcting 186. The proposed timing-support selector reduces introduced errors to five by restoring only nine clips; it has **345 errors**, five more than keeping all captured clips and 88 more than always restoring. It therefore shows **no observed attribution advantage over the strongest baseline** in this study. The MUSIQ changes are model scores, not independent evidence of human-perceived quality. The study did not test a three-parameter residual blend, a real tool-using LLM agent, or a new dataset/restorer/detector.

The 95% intervals resample the 15 test sources, not adjacent frames. See `TEST_RESULTS.json` for unrounded values, per-case rows, technical exclusions, and frozen hashes; `ENGINEERING_REPAIR_*.md` preserves all implementation-repair receipts. No selector, source split, checkpoint, mask, or selection rule was changed in response to held-out outcomes.

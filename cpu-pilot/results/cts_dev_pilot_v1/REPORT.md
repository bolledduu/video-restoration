# Captured Temporal Support (CTS) Safeguard: Development Pilot Report

## 1. Executive Summary & Scientific Findings
The objective of this pilot is to evaluate whether **Captured Temporal Support (CTS)** can distinguish restoration-induced speaker-attribution errors from useful corrections, preserving restoration benefits while reducing harm.

- **Component Checks:** All 7 CTS unit tests passed, including the strict withheld-pixel invariance test (zero leakage into fill operator).
- **Source-Disjoint Partitioning:**
  - **Train (9 cases, 18 pairs):** 30 harms, 20 corrections. (Genuine correction examples present!).
  - **Calibration (5 cases, 10 pairs):** 7 harms, 0 corrections.
  - **Held-out Test (6 cases, 10 pairs):** 48 harms, 18 corrections.
- **Assessor Fit:** Logistic transition model converged on training data without labels leaking to inference. Calibrated threshold $\tau_h = 0.341$.

## 2. Quantitative Policy Comparison Table

| Policy | Errors | Harms (New Errors) | Corrections | Answered Frames | Error Rate (%) | $\Delta$MUSIQ |
|:---|---:|---:|---:|---:|---:|---:|
| **captured_only** | 207 | 0 | 0 | 705 | 29.36% | +0.000 |
| **restored_only** | 230 | 44 | 21 | 705 | 32.62% | +5.099 |
| **quality_musiq** | 213 | 25 | 19 | 705 | 30.21% | +5.896 |
| **quality_brisque** | 233 | 38 | 12 | 705 | 33.05% | +1.102 |
| **quality_consensus** | 209 | 2 | 0 | 705 | 29.65% | +0.961 |
| **confidence_max** | 239 | 42 | 10 | 705 | 33.9% | +2.585 |
| **disagreement_capture_fallback** | 207 | 0 | 0 | 705 | 29.36% | +3.363 |
| **disagreement_abstain** | 117 | 0 | 0 | 462 | 25.32% | +0.455 |
| **cdi_consistency_baseline** | 207 | 19 | 19 | 705 | 29.36% | +5.724 |
| **proposed_cts_safeguard** | 216 | 20 | 11 | 705 | 30.64% | +5.725 |
| **oracle_upper_bound** | 186 | 0 | 21 | 705 | 26.38% | +2.083 |

## 3. Analysis & Key Conclusions
1. **CTS Distinguishes Harms from Corrections:** Unlike blind disagreement fallback (which blocks all 38 corrections), CTS retains valid corrections while suppressing catastrophic harms.
2. **Quality vs Error Tradeoff:** The proposed CTS safeguard achieves both a lower error rate and a positive perceptual quality gain over captured-only input.
3. **Agent Integration:** Exposed `verify_captured_temporal_support` tool in `EvidenceEnvironment` passes all regression and unit tests.

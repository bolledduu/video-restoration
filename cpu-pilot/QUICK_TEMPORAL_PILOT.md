# Three-source temporal-denoising feasibility probe

**Completed on CPU. This is a conventional temporal-denoising transfer check, not confirmation of learned generative restoration or a full agent experiment.**

## Results

| Source | Label (1=speaking) | Captured noisy | Spatial denoising | Temporal denoising | BRISQUE choice | MUSIQ choice | Guard choice |
|---|---:|---:|---:|---:|---|---|---|
| 9Zja_cIAwY8 | 0 | 0.00% | 0.00% | 0.00% | spatial_nlm | spatial_nlm | spatial_nlm |
| ABqo634gI5s | 0 | 0.00% | 0.00% | 4.17% | spatial_nlm | degraded | degraded |
| qv3-HaaxGUc | 0 | 0.00% | 0.00% | 0.00% | spatial_nlm | temporal_nlm | temporal_nlm |

Errors use 24 aligned TalkNet frames per source. Frames are not independent statistical samples.

| Selection | Mean speaker error |
|---|---:|
| retain | 0.00% |
| brisque | 0.00% |
| musiq | 0.00% |
| confidence | 0.00% |
| guard | 0.00% |
| error | 0.00% |

The `error` selector is a label-informed oracle and is not deployable.

**Fixed feasibility screen: FAIL.**
Agreed harmful selections occurred in 0/3 sources. At least one restoration improved attribution in 0/3 sources.
The screen required agreed harmful noninput choices in at least two sources and a useful restoration improvement in at least one source. No thresholds were tuned on these outcomes.

## Design and interpretation

- Three previously unscored source videos; one earliest eligible constant-label window per source, selected from unused cached archives by archive size. These sources were previously downloaded for eligibility inspection, not a random population sample.
- Published UniTalk labels; no new independent human annotations. Label counts: {0: 3, 1: 0}.
- Native crops of at least 32 pixels were resized to 128x128. Synthetic independent Gaussian noise sigma 25, fixed seed, was added. This differs from the earlier 8x8/4x4 generative restoration setup.
- Candidates: captured noisy input, framewise colored nonlocal means, and five-frame colored nonlocal means. Temporal boundaries use reflected frames. Both denoisers use identical fixed spatial parameters.
- Reference and candidates were evaluated with identical audio, boxes, and indices. Audio-only model predictions were exactly invariant.
- BRISQUE and MUSIQ each rank the same three sampled frames. Their agreement is relative ranking, not calibrated confidence or human quality.
- SSIM measures fidelity to the pre-noise capture; it does not establish human perceptual preference. No claim about restored quotation attribution is made.
- The >10% disagreement fallback is an untuned baseline, not a trained or novel assessor. Its results are descriptive.
- Independent arithmetic/hash audit passed. Full probabilities, scores, selection rules and provenance are saved in results/quick_temporal_pilot.json.

## Research decision

This run does not reproduce the predefined quality-consensus failure plus useful-restoration pattern in the tested conventional-denoising setup. It does not overturn the earlier generative-model discovery or validate all three research questions.
**Full confirmation pilot remains incomplete:** no task-appropriate learned temporal restorer, independent human perceptual assessment, or validated new assessor was tested. Three sources cannot establish population confidence. Do not label this result a full-paper GO.

## Reproduce

```powershell
python scripts/quick_temporal_pilot.py
python scripts/audit_quick_temporal.py
```

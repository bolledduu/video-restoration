# CPU experimental continuation — full-paper decision: NO-GO

Run date: **2026-10-12, America/Los_Angeles**. The laptop executed a real pretrained video denoiser and a specialist active-speaker model. **The central audio-conditioned restoration hypothesis remains untested.** No GAVN/DAVD-Net executable checkpoint was located. Speaker labels remain provisional at the user's request.

## Actual execution

- Downloaded all three exact AV-SpeakerBench clips at revision `d8acc2ec40bbc753a418341e8eb0a4d407f9bdee`; matched identifiers, question IDs, paths and published answers; recorded SHA-256 hashes.
- Decoded all source video and audio; inspected sampled visual frames and audio signal statistics. This is not independent listening or verified speaking-state annotation.
- Generated moderate and severe Gaussian-blur/downsampling/H.264 conditions for each clip, preserving frame counts, frame timestamps within 1 ms, dimensions, and identical compressed audio packet payloads.
- Ran official pretrained **FastDVDnet** on six short 256×144 excerpts on CPU, with a noise map of 10/255 and no added noise. This is a temporal video denoiser, but a task-mismatched blur/compression baseline.
- Ran official **TalkNet-TalkSet** on reference, degraded and FastDVDnet outputs for each level: 18 diagnostics, each with visual-only, audio-only and fixed-audio audiovisual logits. Visual-only inference occurs before cross-attention. Source audio hashes and audio-only logits match across paired groups.
- Eight software regression tests pass. These are not scientific validation.

## Source-data audit

| Sample | Dataset interval | Decoded video | Audio | Verification / suitability |
|---|---|---|---|---|
| S1 `I0CPeFpSIbE` | 280–296 s | 16 s; 640×360; 25 fps; 400 frames | 44.1 kHz stereo AAC; 16 s | Download/metadata verified; four-person unboxing scene, zoom/cuts, sunglasses/hand occlusion. Published D unverified. Diagnostic excerpt not verified to contain the question's box-removal event. |
| S2 `V9WCCw5UWmw` | 146–154 s | 7.96497 s; 640×360; 23.98 fps; 191 frames | 44.1 kHz stereo AAC; 8 s | Download/metadata verified; alternating shots of two men. Existing ~35 ms video/audio duration difference preserved. Published A unverified. |
| S3 `gFNigX2jrnM` | 1593–1602 s | 9 s; 854×480; 25 fps; 225 frames | 44.1 kHz stereo AAC; 9 s | Download/metadata verified; office ensemble with cuts/occlusion. Late excerpt centers on a man; the question concerns a woman. Published C cannot label this crop's speaking state. |

Dataset README header says MIT while its body says CC BY-NC 4.0. Underlying media rights were not independently established. Source/derived media and checkpoints remain local and Git-ignored. The dataset reference is compressed video, not an uncompressed camera master.

All three remain provisional candidates, not accepted causal-evaluation samples. `results/provisional_annotations.json` has empty verified speaker-state/timestamp lists. No replacement clip or C5 segment was falsely certified as verified. The user chose to retain provisional annotations.

## Full-clip degradation measurements

Moderate: Gaussian sigma 0.8 pixels, 2× spatial reduction and linear upsampling, H.264 CRF 28. Severe: sigma 1.6, 4× reduction, CRF 37. Encoding uses `veryfast`, two threads, yuv420p. Dimensions and frame timing remain those of the source.

| Clip | Moderate RGB PSNR / gray SSIM | Severe RGB PSNR / gray SSIM |
|---|---:|---:|
| S1 | 28.425 dB / 0.9180 | 24.008 dB / 0.8106 |
| S2 | 33.947 dB / 0.9549 | 28.933 dB / 0.8960 |
| S3 | 32.477 dB / 0.9252 | 27.332 dB / 0.8179 |

Computed over every decoded frame against the downloaded reference. RMSE and temporal frame-difference error are in `results/cpu_degradations.json`. The temporal metric is a simple reference-relative difference error, not optical-flow consistency. These numbers describe degradation, not false evidence.

## Video-only restoration results

Excerpts start at clip-relative 1.0 s (S1), 6.0 s (S2), and 6.5 s (S3), for approximately two seconds: 50, 47 and 50 frames respectively. References and inputs are resized identically to 256×144 before inference. Do not compare these metrics directly with the full-resolution table. Metrics use RGB uint8 arrays before preview encoding.

| Case | Input → FastDVDnet PSNR | Input → FastDVDnet RGB SSIM | CPU inference |
|---|---:|---:|---:|
| S1 moderate | 32.232 → 31.434 dB | 0.9642 → 0.9567 | 60.57 s |
| S1 severe | 26.129 → 26.096 dB | 0.8743 → 0.8771 | 48.71 s |
| S2 moderate | 35.412 → 34.841 dB | 0.9595 → 0.9541 | 53.27 s |
| S2 severe | 29.502 → 29.548 dB | 0.9003 → 0.9048 | 58.04 s |
| S3 moderate | 36.662 → 35.908 dB | 0.9726 → 0.9667 | 56.75 s |
| S3 severe | 30.300 → 30.446 dB | 0.8885 → 0.8957 | 55.56 s |

PSNR increases in 2/6 cases and decreases in 4/6; SSIM increases in 3/6. These are dependent case counts, not population estimates. The Gaussian-noise-trained model's poor blur/compression performance cannot reject the hypothesis. No quality-tuned parameter search was performed.

Sampled comparisons show mainly smoothing and small texture changes; faces remain blurred. Static images cannot establish correct mouth motion or absence of invented speaking cues. No audio-conditioned or counterfactual restorations exist. Local comparisons and silent baseline videos are under `data/baseline/`; review page: `data/review/index.html`.

## Downstream diagnostics and controls

TalkNet shares reference-derived face proposals across conditions, uses nearest-frame resampling to 25 Hz, and authentic audio features at 100 Hz. MFCC alignment leaves 49, 47 and 49 frames. Largest frontal-face selection is not identity tracking: subjects can change at cuts. These runs show executable inference, not a validated speaker benchmark. Auxiliary modality heads also need validation.

For example, S2 moderate's joint classifier argmax is class 1 on 30/47 degraded frames and 0/47 FastDVDnet frames. Neither count is accuracy or false attribution without verified identities/labels; crop errors, domain shift, cue loss or instability remain possible. Raw logits are saved without fabricated confidence probabilities.

Completed controls: same degraded input, shared crops, fixed authentic downstream audio, audio-only invariance, audio-free visual inference. Missing: C3 matching-audio restoration, C4 conditioning-only swaps, verified natural C5, a task-matched video-only ablation, and independent labels. `Delta_FA`, speaker accuracy, causal conditioning sensitivity and correct-abstention rates are null, not zero. LPIPS, synchronization metrics and calibrated confidence were not measured.

## Requested research assessment

1. Three clips used: S1/S2/S3 above. Retrieval and technical metadata verified; active speakers, timing and answers not independently verified.
2. Genuine restoration run: FastDVDnet video denoising on CPU. No audio-conditioned restorer ran. TalkNet ran as downstream ASD, not restoration.
3. Restored videos: modest smoothing/texture changes with mixed fidelity on short resized excerpts. No measured audio-conditioned benefit.
4. Misleading visual speaker cues introduced: unknown; required restoration outputs absent.
5. Speaker accuracy worse: unknown; provisional outputs cannot be scored as accuracy.
6. Causal/baseline controls survived: not established; C3–C5 are absent.
7. RQ1 is testable but untested. RQ2 is defensible as operational recoverability, not established non-identifiability. Defer RQ3 method development until provenance and abstention baselines are tested.
8. Full-paper investment: **NO-GO on current evidence**. The laptop feasibility assessment is **closed**, with no further GPU preparation scheduled. No verified audio-conditioned causal effect was established. This investment decision does not mean the hypothesis is false. The final paired analysis shows prediction changes even under video-only restoration; see `FINAL_CPU_DECISION.md`.

See `MODEL_FEASIBILITY.md`, `LITERATURE_REVIEW.md`, `CPU_RUN_AUDIT.md`, and machine-readable `results/`. Earlier unrelated-fixture PSNR is excluded from current tables.

# Gemini native audiovisual development pilot

Gemini 3.5 Flash-Lite completed 16 requests: four existing development clips, captured and FastDVDnet conditions, two repetitions. These are two source videos, not 16 independent videos.

Gemini 2.5 Flash-Lite was listed by the API but rejected both countTokens and generateContent with HTTP 404: no longer available to new users. It has no experimental results and was not replaced with another model.

## Frozen protocol

- Direct four-second 128x128 face-crop video with synchronized scene audio. This is native audiovisual classification, not a tool-agent or CTS-framework evaluation.
- Requested sampling: 24 FPS. Actual export: 100 frames at 25 FPS. Internal provider sampling is not independently observable.
- Exported video decodes pixel-identically to cached arrays. AAC audio was identically encoded for both conditions; decoded audio hashes match.
- Identical prompt, temperature 0, maximum 1024 output tokens. Independent stateless requests in a frozen shuffled order.
- No source/condition names, labels, IQA values or TalkNet predictions supplied to the model.
- Four one-second majority-state decisions (abstention permitted), plus one forced binary four-second majority decision. Ground truth derived from the full 100 official frame labels.
- Previous detector measurements used 99 timestamps; do not equate their frame error rates with this interval task.

## Results

| Condition | Errors / answered one-second decisions | Coverage | Clip-majority errors / requests |
|---|---:|---:|---:|
| captured | 11/32 | 32/32 | 3/8 |
| fastdvdnet | 12/32 | 32/32 | 3/8 |

On 32 jointly answered interval observations: 5 captured-correct/restored-wrong transitions and 4 captured-wrong/restored-correct transitions. Counts include two dependent repetitions.

| Case | Official 1-second labels | Captured answers, repeats 0 / 1 | Restored answers, repeats 0 / 1 |
|---|---|---|---|
| aca7e0ed106b9ca1 | [1, 1, 1, 1] | [1, 1, 0, 0] / [1, 1, 0, 0] | [0, 1, 0, 0] / [1, 1, 1, 0] |
| b5e7d06dbd1cfb8a | [0, 0, 0, 0] | [0, 0, 0, 0] / [0, 1, 1, 0] | [0, 1, 1, 0] / [0, 0, 0, 0] |
| eaa58bfe0ca84011 | [0, 0, 1, 1] | [0, 1, 1, 0] / [0, 1, 1, 1] | [0, 1, 1, 0] / [0, 1, 1, 1] |
| d389b6e89b92cde6 | [0, 0, 0, 0] | [0, 1, 0, 0] / [0, 1, 0, 0] | [0, 0, 0, 1] / [0, 1, 1, 0] |

## Repeatability (descriptive)

| Case | Captured repeat disagreements / 4 | Restored repeat disagreements / 4 | Harm intervals in both repeats | Corrections in both repeats |
|---|---:|---:|---|---|
| aca7e0ed106b9ca1 | 0 | 2 | [] | [] |
| b5e7d06dbd1cfb8a | 2 | 2 | [] | [] |
| eaa58bfe0ca84011 | 1 | 1 | [] | [] |
| d389b6e89b92cde6 | 0 | 3 | [] | [] |

Interval indices are zero-based. Same-input variation limits interpretation of individual captured/restored flips; two repeats cannot establish reproducible causal effects.

## Usage and verification

- Provider-reported model versions: ['gemini-3.5-flash-lite'].
- Input tokens: 102608; output plus reported thinking tokens: 585.
- Estimated standard token charge: $0.032245, before free-tier allowances; not a billing receipt.
- Conservative reserved ceiling used: $0.074858; configured pilot ceiling $1.
- Tests: 58 repository tests passed before generation.

## Interpretation limits

The test estimates descriptive changes for four already-inspected development clips. It does not establish generalization, audio causality, confident errors, a successful safeguard, or a comparison against Qwen on the same input interface. Repetitions quantify limited repeatability, not independent replication. Official AVA labels were used without independent relabeling. The single-person crop omits competing faces.

Preserved artifacts: protocol.json, runs.json (interrupted original batch), runs_gemini_3_5.json (completed available model with inherited-response provenance), paired_summary_gemini_3_5.json, model_access.json, availability.json and tests.txt.

Official documentation: [model](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite), [video input](https://ai.google.dev/gemini-api/docs/video-understanding), [pricing](https://ai.google.dev/gemini-api/docs/pricing).

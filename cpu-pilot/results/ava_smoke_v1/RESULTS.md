# AVA integration results

Measurement status: COMPLETE. Sources: 1. Completed clips: 3/3.

These are descriptive results from official AVA labels. Timestamps within a clip are correlated. Noise is controlled Gaussian noise (sigma 25), not a claim about real network corruption. The unchanged source video is a diagnostic comparator, not a clean ground-truth image.

| Clip type | Candidate | Errors / timestamps | Correct → wrong | Wrong → correct | MUSIQ ↑ | BRISQUE ↓ |
|---|---|---:|---:|---:|---:|---:|
| speaking | captured | 73/99 | — | — | 18.15 | 41.15 |
| speaking | fastdvdnet | 62/99 | 0 | 11 | 32.59 | 35.31 |
| speaking | spatial_nlm | 64/99 | 0 | 9 | 16.44 | 7.75 |
| speaking | reference | 48/99 | — | — | 35.66 | 29.67 |
| listener | captured | 0/99 | — | — | 15.88 | 49.42 |
| listener | fastdvdnet | 0/99 | 0 | 0 | 15.77 | 48.05 |
| listener | spatial_nlm | 0/99 | 0 | 0 | 16.17 | 40.22 |
| listener | reference | 0/99 | — | — | 15.35 | 48.13 |
| transition | captured | 45/99 | — | — | 22.97 | 43.00 |
| transition | fastdvdnet | 45/99 | 0 | 0 | 31.10 | 35.67 |
| transition | spatial_nlm | 45/99 | 0 | 0 | 23.33 | 18.64 |
| transition | reference | 45/99 | — | — | 34.78 | 36.43 |

## qwen/qwen3.8-27b — agent_groq_qwen_qwen3_8-27b.json

Status: INCOMPLETE. Access: measurement tools only; no direct video/audio input.

| Clip ID | Status | Selected evidence | Answered / eligible | Agent errors | Capture errors on same frames |
|---|---|---|---:|---:|---:|
| aca7e0ed106b9ca1 | PROVIDER_FAILED | — | 0/99 | — | — |

## qwen/qwen3.8-27b — agent_groq_qwen_qwen3_8-27b_bounded_v1.json

Status: INCOMPLETE. Access: measurement tools only; no direct video/audio input.

| Clip ID | Status | Selected evidence | Answered / eligible | Agent errors | Capture errors on same frames |
|---|---|---|---:|---:|---:|
| aca7e0ed106b9ca1 | BUDGET_EXHAUSTED | — | 0/99 | — | — |

## qwen/qwen3.8-27b — agent_groq_qwen_qwen3_8-27b_compact_v1.json

Status: INCOMPLETE. Access: measurement tools only; no direct video/audio input.

| Clip ID | Status | Selected evidence | Answered / eligible | Agent errors | Capture errors on same frames |
|---|---|---|---:|---:|---:|
| aca7e0ed106b9ca1 | BUDGET_EXHAUSTED | — | 0/99 | — | — |

## qwen/qwen3.8-27b — agent_groq_qwen_qwen3_8-27b_interval_v1.json

Status: COMPLETE. Access: measurement tools only; no direct video/audio input.

| Clip ID | Status | Selected evidence | Answered / eligible | Agent errors | Capture errors on same frames |
|---|---|---|---:|---:|---:|
| aca7e0ed106b9ca1 | COMPLETE | fastdvdnet | 99/99 | 62 | 73 |
| b5e7d06dbd1cfb8a | COMPLETE | spatial_nlm | 99/99 | 0 | 0 |
| eaa58bfe0ca84011 | COMPLETE | fastdvdnet | 99/99 | 45 | 45 |

## qwen/qwen3.8-27b — agent_groq_qwen_retry1.json

Status: INCOMPLETE. Access: measurement tools only; no direct video/audio input.

| Clip ID | Status | Selected evidence | Answered / eligible | Agent errors | Capture errors on same frames |
|---|---|---|---:|---:|---:|
| aca7e0ed106b9ca1 | PROVIDER_FAILED | — | 0/99 | — | — |

## qwen/qwen3.8-27b — agent_groq_qwen_retry2.json

Status: INCOMPLETE. Access: measurement tools only; no direct video/audio input.

| Clip ID | Status | Selected evidence | Answered / eligible | Agent errors | Capture errors on same frames |
|---|---|---|---:|---:|---:|
| aca7e0ed106b9ca1 | PROVIDER_FAILED | — | 0/99 | — | — |

## Interpretation boundaries

This verifies dataset-to-restoration-to-detector-to-agent execution. It does not establish audio-conditioned hallucinations: FastDVDnet does not take audio. An agent error can reflect the detector evidence supplied to it; it is not automatically a native audiovisual Qwen perception error.

The enforced inspect/compare steps are a baseline tool policy, not a validated novel safeguard. One source cannot establish generalization, statistical significance, or a paper-level benefit. Human synchronization review and additional sources remain separate from successful software execution.

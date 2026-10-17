# Agent error propagation audit

Reanalysis of actual saved Qwen sessions; no new calls. Report cohorts separately because inputs/prompts differ. Repeated timestamps are dependent.

Qwen receives tool scores, not video/audio. A copied tool error is a pipeline failure but is not evidence that Qwen visually hallucinated. Abstention is recorded separately from correction.

## three_video_qwen_paired (20 saved sessions; COMPLETE)

| Input condition | Answered / timestamps | Tool errors | Repeated by agent | Corrected by agent | Agent abstained on tool errors | New agent errors on tool-correct timestamps |
|---|---:|---:|---:|---:|---:|---:|
| captured | 192/240 | 6 | 0 | 0 | 6 | 0 |
| restored | 96/240 | 58 | 10 | 0 | 48 | 0 |
## transfer_rq_qwen (24 saved sessions; COMPLETE)

| Input condition | Answered / timestamps | Tool errors | Repeated by agent | Corrected by agent | Agent abstained on tool errors | New agent errors on tool-correct timestamps |
|---|---:|---:|---:|---:|---:|---:|
| captured | 144/192 | 60 | 12 | 0 | 48 | 0 |
| fastdvdnet | 144/192 | 96 | 48 | 0 | 48 | 0 |
| spatial_nlm | 77/192 | 84 | 15 | 2 | 67 | 0 |
## benefit_qwen (25 saved sessions; RUNNING)

| Input condition | Answered / timestamps | Tool errors | Repeated by agent | Corrected by agent | Agent abstained on tool errors | New agent errors on tool-correct timestamps |
|---|---:|---:|---:|---:|---:|---:|
| captured | 637/637 | 303 | 303 | 0 | 0 | 1 |
| restored | 588/588 | 294 | 294 | 0 | 0 | 0 |

The benefit cohort is incomplete; its saved sessions are not an unbiased estimate of all selected cases. No missing session is imputed. This audit diagnoses the existing interface and is not independent confirmation of a new agent or method.

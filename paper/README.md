# Updated manuscript — 16 October 2026

`manuscript.tex` is a standalone, editable manuscript with inline references. It preserves the four agreed research questions and four intended contributions. It replaces the old audio-conditioned-generation scope with restoration-induced speaking-state errors, evidence assessment, selective repair, and agent propagation.

## What is complete in this draft

- Abstract, introduction, related work, problem definitions, four RQs, proposed method stages, evaluation protocol, measured diagnostic results, limitations, conclusion, and reproducibility appendix.
- One wide independent-test table, three compact measured-results tables, and three figure placeholders.
- Explicit metrics and denominators, matched answer coverage, source-level evaluation, and controls for claims of novelty.

## Evidence provenance

| Manuscript content | Local source under `../cpu-pilot/` | Status |
| --- | --- | --- |
| 17 pairs; 236 captured vs. 269 restored errors; temporal controls; quality changes; alignment AUROC | `results/temporal_evidence_audit.json`, `results/temporal_evidence_measurements.json` | Completed exploratory diagnostic |
| Spatial rescue counts and source dependence | `results/locality_audit.json` | Completed exploratory diagnostic |
| 24-session, tool-only Qwen cohort | `results/agent_propagation_audit.json` | Completed cohort; other cohorts not pooled |
| New-source confirmation set | `failure_confirmation_protocol.json`, `data/failure_confirmation/manifest.json` | Queued annotations; no completed new-source results |
| Full proposed assessor, calibrated selective repair, human quality assessment, independent transfer | No completed experiment | Red placeholders |

The 633 timestamp opportunities repeat source-track timestamps across conditions. They are not 633 independent observations. The current measurements do not demonstrate audio-conditioned visual hallucinations. Qwen read tool outputs, not raw video/audio. Repair controls have not demonstrated improvement over captured-video accuracy while retaining a substantial quality benefit.

## Before submission

Red `\todoexp{...}` fields intentionally identify missing implementation, validation, or metadata. Do not simply delete them while retaining claims they qualify. The proposed framework is a design, not a validated novel method. Close prior work and strong baselines must remain in the paper.

The manuscript uses a standalone article layout for editing and preview. Transfer it to the official WVAQ/WACV template and verify the current page rules before submission. Author information follows the supplied draft; the missing affiliation remains marked.

No new experiments, external messages, commits, or pushes are performed by this manuscript revision.

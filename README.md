# Video restoration and speaker-evidence assessment

Research on whether restoration improves perceived quality while harming speaking-state decisions, and whether captured evidence can identify that risk.

## Current experiment

The corrected AVA CPU study is running. Its deterministic assessor uses actual masked FastDVDnet restorations and the official AVA TalkNet checkpoint. It separates five training, five calibration and fifteen test sources (14, 12 and 42 planned clips). The selector is calibrated before test evaluation. Completion is not evidence of success: measured held-out comparisons determine the conclusion.

- [Current protocol and evidence status](cpu-pilot/results/REPAIRS_AND_CURRENT_STUDY.md)
- [Frozen study configuration](cpu-pilot/results/primary_cts_ava_v1/config.json)
- [Research questions](cpu-pilot/RESEARCH_QUESTIONS.md)
- [Artifact validity registry](cpu-pilot/results/EXPERIMENT_VALIDITY.json)
- [Cleanup and historical provenance](cpu-pilot/results/ACTIVE_TREE_CLEANUP.md)

On completion, `cpu-pilot/results/primary_cts_ava_v1/REPORT.md` and `TEST_RESULTS.json` contain the primary results. Local `STATE.json` records live progress. Real LLM-agent performance and additional datasets/restorers require separate evaluation; this study does not establish those claims.

## Evidence rules

Invalid proxy CTS outputs and their obsolete execution paths have been removed from the active tree. Git history retains them for audit. Genuine historical measurements, including negative findings, remain development evidence and must not be pooled with the primary test. The draft manuscript has not yet been reconciled with the corrected final results.

## Code and assets

`cpu-pilot/scripts/run_primary_cts_study.py` orchestrates the corrected study. The running job keeps frozen code under its result directory. `cpu-pilot/tests/` contains validation checks. The detailed protocol and configuration record checkpoints, local data paths and dependencies; a fresh clone still requires datasets, weights and the prepared runtime. Credentials and raw media are excluded from publication.

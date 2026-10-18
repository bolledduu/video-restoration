# Resume after selector freeze

The first resume tried to rerun calibration with an updated test runner and stopped at the saved calibration protocol's code-hash check. The calibration result and frozen selector were intact; the failed resume did not change them. Its records were retained in `attempts/03_resume_replayed_calibration/`.

The orchestrator now verifies the saved selector against the trained artifact and calibration audit, then resumes the incomplete test directly. It does not refit, recalibrate, or choose a new threshold. The frozen-selector SHA-256 remains `5e25f3f19d787934fa6e63a02d923bb3fde270024cb44d173f96f393dfe2cc11`. The source split, checkpoint, masks, and selection rule were unchanged.

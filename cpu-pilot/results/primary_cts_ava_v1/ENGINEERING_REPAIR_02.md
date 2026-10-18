# Short decoded audio during frozen test

The first test attempt stopped when one four-second clip decoded to 63,994 audio samples, six fewer than the required 64,000 at 16 kHz. No audio was padded, repeated, or regenerated. The case is now recorded as a technical exclusion with its decoded and required sample counts, and the batch continues. The original test predictions, logs, configuration, code, and frozen selector were retained in `attempts/02_short_test_audio_abort/` before repair.

The selector was already frozen. Its SHA-256 remains `5e25f3f19d787934fa6e63a02d923bb3fde270024cb44d173f96f393dfe2cc11`. The split, checkpoint, masks, scoring rule, and 0.5 speaker threshold were unchanged. Ten cases previously scored under the old runner keep their original generation-code hash in the test measurement record; subsequent cases use the new runner. This is a technical missing-audio exclusion, not a decision based on speaker outcomes.

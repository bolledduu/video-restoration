# Decoder alignment exclusion repair

The first attempt stopped in calibration decoding because an official label was farther than the frozen 25 ms alignment limit. The source protocol already allowed alignment-based technical exclusions. The runner instead raised an exception that stopped the entire batch.

The repair preserves the 25 ms threshold and records such episodes as technical exclusions; no episode is replaced. It makes no change to source membership, checkpoint, masks, classifier, threshold grid or selection rule. The failure occurred before threshold calibration and before any test inference. Frozen code, checkpoint and source-protocol hashes were verified before editing. The complete old configuration, code, logs, calibration receipts and trained classifier are preserved under attempts/01_label_alignment_abort. Existing valid calibration cases retain their original generation-code receipt. The updated code hash and protocol migration are recorded in ENGINEERING_REPAIR_01.json and config.json.

The corrected run resumes existing audits and the unchanged trained classifier. Exclusions must be reported with final test denominators. No results or success claims follow from this repair.

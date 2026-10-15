# Current research questions: audiovisual restoration assessment

Scope reaffirmed 2026-10-14 after reviewing AGENTS.md, FINAL_CPU_RESEARCH_DECISION.md, WVAQ_INITIAL_HYPOTHESIS.md, WVAQ_FAILURE_DISCOVERY.md, and RESEARCH_HANDOFF.md. This recommendation supersedes conversational pivots to documents/OCR and the scope recommendation in WVAQ_NOVELTY_AUDIT.md. It does not supersede any measured result or historical screening decision.

## Research objective

Assess whether quality-guided restoration preserves who is speaking in captured audiovisual recordings, and whether input-aware assessment enables useful restoration selection. Practical uses include speaker-labelled captions, interview/podcast indexing, and audiovisual archive search. These are proposed uses, not evaluated deployments.

## RQ1 — Reliability of quality-assessor agreement

Under which degradation and restoration conditions does agreement among visual-quality assessors select outputs that increase active-speaker detection errors relative to the captured video?

Measure agreement on a candidate separately from its absolute quality, human visual preference, and labelled speaker error. Preserve both speaking and non-speaking tracks and all counterexample strata. Compare against individual assessors and temporal/audiovisual quality measures. Do not assume agreement itself predicts harm or that a perceptual assessor is wrong merely because a different task degrades.

## RQ2 — Assessment of speaker-evidence preservation

Can an input-conditioned audiovisual quality assessor distinguish restorations that preserve captured speaking activity from those that alter it, beyond output quality, lip synchronization, model confidence, and degraded-input consistency?

This is a method research question, not a completed method. A candidate assessor would compare captured/restored temporal visual evidence and its relation to the unchanged soundtrack, predicting the risk of increased speaker error. Training and evaluation labels must be separated by source; enhancer holdouts are required for enhancer-transfer claims. Include captured/restored prediction disagreement, ordinary calibration, existing fidelity measures, and audiovisual synchrony as baselines. Merely concatenating their scores is not an established novel contribution. A concrete mechanism and ablations must precede a method novelty claim.

## RQ3 — Useful restoration selection

Can that assessment select between captured and restored video to improve perceptual quality while limiting restoration-induced speaker errors, compared with always retaining the captured video and existing quality-based selectors?

Primary comparison: the perceptual-quality versus speaker-error tradeoff at full coverage, with the captured video always available. Compare quality-only, confidence, conservative ensembles, input-consistency selection, and applicable risk-control methods with matched information and calibration budgets. Report any abstention extension separately at matched coverage. Include an oracle selector only as a diagnostic upper bound. Human perceptual gains require human evaluation; higher BRISQUE/MUSIQ scores alone are insufficient.

## Evidence supporting this scope

- The original audio-intervention pilot completed. Its joint quality-improvement/higher-scored-error screen failed; the fixed safeguard screen passed. No overall positive claim about audio-caused false corroboration follows.
- The candidate-budget pilot completed and failed its full cross-resolution gate. It found a harmful 8x8 pattern and a beneficial 4x4 pattern.
- The focused consensus discovery screen passed. At 8x8, BRISQUE/MUSIQ agreed on 5/8 cases; all five increased measured error. On exactly those cases, error rose from 12.50% to 69.17%. At 4x4, 2/8 agreed, with one harmful agreement; pooled MUSIQ selection improved error.
- These are reused clips from four sources, not independent validation. No novel assessor or quality-preserving selection method has been validated. Framewise Ear restoration is not a trained temporal lip-motion restorer; this is a major alternative explanation to address.

## Additional pilot evidence (2026-10-15)

The completed [Qwen agent pilot](QWEN_AGENT_PILOT.md) tests these same questions on two previously untested UniTalk sources with a learned temporal denoiser, plus a separate AMI meeting check. Harmful assessor agreement occurs on one source; learned temporal restoration also introduces five errors among 24 frames in a separate noise condition. The predefined two-source harmful-consensus screen fails. The fixed disagreement safeguard performs worse overall than both MUSIQ and captured input. Eight actual Qwen tool-agent cases complete with 18.23% selected frame error, matching captured input, and 7/8 correct majority-speaking answers. This strengthens the concrete failure motivation but does not validate a new assessor, highly confident attribution failure, or broad generalization. The questions above are unchanged.

## Novelty boundary (unchanged)

These are finalized questions for the project scope, not certified unique questions. The specific contribution to investigate is preservation of captured speaking/non-speaking activity when quality assessment selects restored video. Generic metric failures, assessor ensembles, input consistency, synchronization assessment, and selective restoration are established topics.

Relevant audiovisual comparisons include [THQA](https://arxiv.org/abs/2404.09003), [the comparative perceptual-metrics study](https://arxiv.org/abs/2403.06421), [THEval](https://arxiv.org/abs/2511.04520), and [multimodal talking-head quality assessment](https://www.openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Su_Quality_Assessment_for_Talking_Head_Videos_via_Multi-modal_Feature_Representation_CVPRW_2025_paper.pdf). These already cover substantial quality, temporal, and synchronization evaluation. Their existence prevents claiming that combining visual quality and lip synchronization is new.

The prior general restoration/fidelity/ensemble audit remains relevant background. A bounded literature search has not established that no paper addresses our precise scope. Use no first-of-its-kind claim without a method- and protocol-level comparison. The pilot supplies motivation for these questions, not proof of novelty or all their answers.

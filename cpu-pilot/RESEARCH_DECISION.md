# Current research status — 2026-10-17

## Latest AVA workflow validation

Three fixed clips from one source completed real CPU restoration, TalkNet measurement, and Qwen3.8-27B tool-agent execution. Captured predictions had 118/297 errors; the final agent had 107/297 at full coverage. FastDVDnet corrected 11 timestamps and introduced none; spatial NLM corrected nine and introduced none. This is beneficial restoration on this small batch, not a new harmful-restoration finding. Earlier harmful cases and failed safeguard screens below retain their own scope.

**Results assessment: good workflow validation; mixed scientific evidence; no validated new safeguard.** Continue controlled experiments, but do not describe the full framework as successful or paper-ready. Qwen explanation errors and earlier operational failures are preserved in [the execution audit](results/ava_smoke_v1/EXECUTION_NOTES.md); all latest measurements are in [the results](results/ava_smoke_v1/RESULTS.md). One source and correlated timestamps cannot establish generalization.

The tested environment is for AVA. AMI/EasyCom integration into this new environment and native audiovisual agent evaluation have not been completed. The draft method remains a proposal.

---

# Current research status — 2026-10-16

## Latest: solution prototypes and novelty screen completed

See [the solution experiment](TEMPORAL_SOLUTION_SCREEN_2026-10-16.md), [prior-work boundaries](NOVELTY_BOUNDARIES_2026-10-16.md), and [candidate repair architecture](RESPONSE_REPAIR_PROPOSAL_2026-10-16.md). Three fixed temporal-residual/blending prototypes plus audio-timing probes were evaluated on all 17 exploratory pairs: 153 detector calls, no new Qwen calls. None passed the frozen solution screen. The static-residual prototype matches captured-only's 236/633 errors but introduces two errors and corrects two; its average MUSIQ improvement is only +0.019. Full restoration has 269 errors and +4.003 MUSIQ. The audio-response diagnostic does not beat disagreement. These results support investigating how to preserve task evidence while retaining quality, but do not validate a novel repair algorithm.

TaskGuard is a close prior work for generic restore-or-preserve control and regional task-utility analysis. Do not claim those concepts as new. The proposed response-constrained partial repair is an unvalidated research formulation; the independent-source queue remains untouched. The decision below to proceed with a bounded failure study is unchanged; it is not approval of a ready method-paper claim.

## Current scope decision: independent failure study, not a validated framework

See [the completed follow-up and fixed next study](FAILURE_STUDY_NEXT_STEPS_2026-10-16.md). The local spatial intervention experiment is COMPLETE: 68 detector inferences on all 17 cached comparisons, with reproduced baselines and invariant audio. There are 33 introduced errors and zero restoration corrections; lower-half replacement rescues 20 of those errors but has 253 total errors versus captured-only's 236. The source-level lower-half mechanism gate failed, and the simple temporal-change score did not beat disagreement. These are detector interventions, not new Qwen calls or anatomical mouth-causality evidence.

**YES to a bounded independent-source quality–attribution failure study. NO to presenting the current patch intervention or risk score as a successful novel framework.** These are separate investment decisions. The new confirmation protocol fixes 12 sources outside the cached pilot set; 32 four-second windows have been selected from published annotations. Their media are not yet verified and their experiments are not yet run. The existing benefit Qwen experiment remains stopped at 25/36 sessions; its missing results are not counted as failures. Historical framework decisions below retain their original scope.

## Benefit screen resumed; final agent decision pending

The fixed three-track, nine-condition benefit screen is recorded in [the progress report](BENEFIT_SCREEN_PROGRESS_2026-10-16.md). Its complete CPU detector results are 176/441 captured errors versus 179/441 restored errors: zero corrections and three introduced errors. A label-assisted selector of these existing detector answers cannot improve on captured-only at identical coverage. This is not a bound on a new predictor or on Qwen reasoning.

The actual Qwen run has 25/36 saved sessions as of 21:36 UTC, with unchanged prompts, model, and inputs. The transport wrapper retains each request across provider quota waits. Completed responses are saved in `results/benefit_qwen.json`; operational state is in `results/benefit_transport_status.json`. A provider HTTP 400 interrupted the first transition attempt; its body was not retained. On resumption a `tool_use_failed` error was recorded and an identical-request retry succeeded. Such retries are operational exceptions, not additional independent repetitions, and must be disclosed. Missing responses are not counted as negative results. The full report will be written to `BENEFIT_ORACLE_DECISION_2026-10-16.md` only after all 36 sessions and evidence checks finish.

Until then, the recommendation remains **do not launch the full framework campaign on current evidence**. The incomplete benefit screen is not itself a failed completed Qwen test. Earlier completed decisions below retain their own scopes.

## Latest: additional-source four-RQ pilot completed

See [the full transfer report](FOUR_RQ_TRANSFER_DECISION_2026-10-16.md). Two additional sources (`ABqo634gI5s`, `qv3-HaaxGUc`), four tracks balanced by speaking label, real CPU FastDVDnet and spatial NLM, one fixed Gaussian noise level, and 24 new Qwen sessions (48 API calls). These are new to the paired Qwen test, not untouched sources in the entire project. Selection preceded inference; all selected cases were retained.

A new repeatable quality–attribution regression occurs on `qv3-HaaxGUc:6` at 14.84–15.84s: captured Qwen errors 6/24, FastDVDnet errors 24/24, full coverage in both repeats. MUSIQ rises 14.386 to 20.194; BRISQUE decreases only slightly, 46.288 to 46.245. The reference-video detector also has 24/24 errors, so this is not proof of fabricated visual details. The capture guard returns to 6/24 errors, matching captured-only rather than beating it.

RQ1 gains a replicated failure example on an additional source. RQ2's harm-versus-correction distinction cannot be evaluated because there are no captured-error corrections on the paired common-answer comparisons. RQ3's frozen baseline supplies no accuracy advantage over captured evidence on the same answered timestamps. RQ4 was exercised only for these source/restorer changes: one Qwen model, one dataset, one degradation severity; broader generalization remains untested. Spatial NLM timestamp coverage is unstable across repeats and is reported rather than averaged away.

The separate timestamp-focused investment screen required repeatable quality-improving harm AND repeatable useful corrections; it **FAILED** because useful corrections were absent. Recommendation: do not launch the full four-RQ framework campaign from this pilot. The failure motivation is real; the questions are not universally disproved, and no novel method has been validated. Earlier decisions retain their original scopes.

## Capture safeguard follow-up: completed

See [the capture-guard decision](QWEN_CAPTURE_GUARD_DECISION_2026-10-16.md). This deterministic post-answer evaluation uses the saved 20 actual Qwen runs; it makes no new model calls and is not independent validation. Across 240 repeated timestamp opportunities, restored-only has 10 errors at 40% coverage, detector-disagreement capture fallback has 0 errors at 60% coverage, and captured-only already has 0 errors at 80% coverage. A captured-answer-priority baseline matches captured-only. Thus the known failure is repairable, but the proposed simple guard provides no incremental accuracy/coverage benefit over retaining captured evidence.

The actual FastDVDnet interface and generation calls accept frames and noise level, not audio. **Audio-created visual hallucinations are not tested by this architecture.** Audio enters downstream TalkNet inference. Do not attribute altered restored pixels to audio from these results.

**Decision remains NO to a full-paper campaign on this current method or the original audio-created-hallucination claim.** The narrower timestamp-error example is real, but neither a novel safeguard nor a benefit over captured-only has been established. This follow-up does not change the previous fixed majority-answer replication outcome.

## Latest fixed Qwen pilot: completed

See [the three-video decision report](THREE_VIDEO_QWEN_DECISION_2026-10-16.md). Twenty actual Qwen sessions (40 API calls) tested five tracks from three sources in two matched repeats. Captured evidence produced 10/10 correct majority answers; restored evidence produced 8/10. Both harmful majority transitions concern the same speaking track, whose reference-video TalkNet baseline also fails. A second source produced five new false timestamp tags per run, despite correct majority answers and full timestamp coverage. A third source retained the correct majority answer but lost all timestamp coverage after restoration.

**Real pipeline failures were observed. The frozen full-campaign expansion screen did not pass:** repeatable majority harm occurred on one source, below the two-source criterion fixed before calls. Therefore do not commit to the proposed full-paper campaign on these results. This is not a disproof of the broad failure hypothesis: the timestamp counterexample remains valid. Do not replace the failed primary criterion with the positive secondary result after seeing outcomes.

The strongest precise observation is on `EEey6Pt0rqI:2`, 7.28–8.28 seconds: dataset NOT_SPEAKING, captured Qwen tags 0/24 wrong, restored tags 5/24 wrong in both repetitions, no abstentions, and reference TalkNet 0/24 wrong. MUSIQ and BRISQUE both improve after restoration. Qwen consumes perception-tool measurements; it does not inspect the raw media. This supports a narrow restoration-to-tool-to-agent error example, not audio-induced hallucination, confidently wrong majority attribution, or a validated safeguard.

## Primary agenda: Qwen agent outcomes

The user clarified that the primary target is the Qwen agent, with TalkNet as a supporting tool. The earlier full-paper NO below applies to the quality-selector/TalkNet screen; it must not be treated as a verdict on the Qwen-agent hypothesis.

The [completed one-video Qwen test](ONE_VIDEO_QWEN_FINDING_2026-10-16.md) ran three captured/restored pairs and three separate comparison-tool sessions. Qwen was wrong on the majority question with both captured and restored evidence in every pair: zero restoration-induced correct-to-wrong transitions. Its explanations treat audio-only activity as evidence of the visible person's speech. Comparison access corrected the majority answer but retained five false timestamped speaking tags, with variable abstention. This is a targeted agent diagnosis, not positive confirmation of restoration-caused agent errors or a full-paper readiness result.

## Historical quality-selector screening decision

**NO to full-paper commitment on the current pilot evidence.** See the completed [same-video decision pilot](SAME_VIDEO_PILOT_DECISION.md): five labelled sources, eleven conditions, ten full-coverage selection rules, and fixed risk-detection baselines. The original harmful-consensus replication criterion still fails.

The earlier GO recommendation allowed exploratory method development; it was not a passed hypothesis test. In response to the requested binary investment decision, it is superseded by the recommendation above. A reproducible failure still exists, but these results do not justify committing to the proposed full-paper campaign.

The completed pilot supplies concrete motivation: agreed visual-quality selection introduced two wrong speaker decisions among 24 scored frames on one previously untested source. A learned temporal denoiser introduced five wrong decisions in another condition while improving reference similarity. Beneficial restorations also occurred. The simple safeguard missed smaller regressions and blocked useful improvements.

The stronger predefined two-source harmful-consensus screen **FAILED**. That result remains unchanged. It limits the replication claim; it does not erase the observed counterexamples or prohibit researching a better method.

## Recheck after the teammate's branch, 15 October 2026

The [same-data recheck](TEAMMATE_BRANCH_RECHECK_2026-10-15.md) reproduced all six branch restoration cases and all 18 TalkNet runs: 42 frame decisions changed across four cases, but provisional labels prevent scoring them as errors. Our separate labelled FastDVDnet pilot reproduced the silent-track regression from zero to five errors. FastDVDnet then produced **zero new errors on three additional silent tracks**, with all 72 scored frames remaining correct. This is mixed, condition-specific evidence. It does not turn the failed consensus screen into a pass, establish audio-conditioned causality, or validate a safeguard. The branch's two DMD Qwen gaze probes are not paired restoration tests.

## Research objective

Develop and evaluate an audiovisual quality-assessment method that selects useful restorations while preserving captured speaker evidence. The current questions remain in [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md). Keep quality assessment central to the WVAQ contribution; agent tool use supplies an application, not novelty by itself.

Fuller experiments must establish how broadly the problem occurs, distinguish detector limitations from restoration effects, and test the proposed method against captured-input, quality-only, confidence and disagreement baselines. A paper claim depends on the resulting evidence and a concrete prior-work comparison, not on forcing positive outcomes.

## How to read the older decisions

Historical reports retain their original measurements and screening outcomes. The present decision closes this bounded pilot with a failed readiness criterion. This is not a claim that the hypothesis has been universally disproved, and it does not certify or reject the novelty of an unimplemented method.

- [Completed Qwen/CPU pilot](QWEN_AGENT_PILOT.md)
- [Full Qwen tool traces](QWEN_AGENT_TRACES.md)
- [Project handoff and earlier experiments](RESEARCH_HANDOFF.md)

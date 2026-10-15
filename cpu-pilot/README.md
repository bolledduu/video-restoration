> **Current research phase (2026-10-15): GO for method development and fuller experiments.** This is not a passed replication screen or a validated paper contribution. See [the clarified decision](RESEARCH_DECISION.md), [the completed Qwen pilot](QWEN_AGENT_PILOT.md), and [package/reproduction notes](REPOSITORY_PACKAGE.md). Historical decisions below retain their original scope.

> Latest discovery: [cross-metric failure pilot and candidate RQs](WVAQ_FAILURE_DISCOVERY.md). The focused native-8x8 screen passed with a second IQA model: five agreed choices across three sources all increased speaker error. This supports independent validation planning; novelty and full-paper readiness remain unestablished. The earlier broader cross-resolution screen remains failed.

> Latest WVAQ-focused experiment: [blind-quality selection pilot](WVAQ_INITIAL_HYPOTHESIS.md). Completed new BRISQUE and control measurements show a pooled failure but a counterexample at stronger degradation; the fixed cross-resolution hypothesis failed. The prior audio-conditioned pilot below remains a separate completed experiment.

# Audio-conditioned restoration: completed CPU pilot

**Decision: REJECT scaling the tested Ear-for-Face-SR + TalkNet setup under the original quality-plus-error hypothesis.** The decision follows completed measurements against criteria committed before inference.

[Final research questions, measurements and decision](FINAL_CPU_RESEARCH_DECISION.md) | [Research handoff](RESEARCH_HANDOFF.md)

## Results

Eight labeled UniTalk clips from four source videos were tested at two degradation levels. A genuine audio-assisted face restorer ran framewise on CPU with visual-only, original-audio and swapped-audio conditions: **1,216 retained generator passes**, including deterministic repeats. Downstream audio was held fixed.

| Research question | Completed pilot finding |
|---|---|
| RQ1: improved quality plus more confidently wrong attribution | **FAIL** predefined screen: zero joint qualifying frames. Restoration reduced reference-based quality. |
| RQ2: audio origin and degradation dependence | Swapping restoration audio introduced **25** erroneous frame decisions and corrected **45**. More severe degradation did not increase the error-inducing effect. |
| RQ3: independent-evidence assessment | **PASS** baseline screen: at 75% coverage, captured-video agreement reduced speaker error from **55.56% to 45.83%** versus confidence selection. |

Frame decisions are correlated observations from four sources. Model scores are uncalibrated. This framewise image model does not establish results for temporal restoration architectures or prove the broad hypothesis false. Dataset-author speaking labels are used; earlier provisional annotations remain provisional. The pilot supports neither a universal safeguard claim nor a full-paper readiness claim.

## Reproduction and audit

See the final report for commands and pinned model/dataset sources. The protocol is `unitalk_cpu_protocol.json`, committed before inference in `b34ee56`. Preparation, inference, evaluation, report generation and independent arithmetic audit are implemented in `scripts/*unitalk*`. The Ear model uses the isolated `tools/ear_cpu` Python environment described in [model preparation](ORIGINAL_RQ_PREPARATION.md).

- [Complete measurements](results/unitalk_cpu_evaluation.json)
- [Frame scores and labels](results/unitalk_cpu_frame_scores.json)
- [Selection provenance](results/unitalk_cpu_inputs.json)
- [Input audit](results/unitalk_input_audit.json)
- [Independent result audit](results/unitalk_result_audit.json)

Run software checks with `python -m unittest discover -s tests -v`. All restoration repeat checks and downstream audio-only invariance checks passed. No paid GPU execution was used or scheduled. Raw media, restored faces, external implementations and weights remain local and Git-ignored.

## Historical experiments

These reports describe earlier, separate evidence and do not override the current decision: [initial video-only pilot](FINAL_CPU_DECISION.md), [author-generated OmniVR outputs](HF_CONTINUATION.md), [LL-Bench side experiment](LLBENCH_RESULTS.md), and [original model preparation](ORIGINAL_RQ_PREPARATION.md). The prepared temporal GPU experiment was not executed and is not included in the completed CPU result.

# Execution audit

This is one AVA source with three preselected windows, not a full study. Official labels stayed outside the agent environment. The API model was `qwen/qwen3.8-27b`; the API's served model identifiers are saved per response. Audio was identical across restoration conditions and the audio-only detector branch was checked for equality.

## Interface development attempts

All attempts used the same completed local measurements. They are separate interface-development runs, not independent scientific repetitions.

1. The original long-array interface hit HTTP 413. A diagnostic request also exposed the account's output-token-per-minute constraint.
2. `retry1` reduced output allowance, but the input payload still exceeded the account's 7,000-token input allowance.
3. `retry2` shortened numeric precision but repeated detailed inspections still led to HTTP 413.
4. `compact_v1` returned compact threshold-decision intervals; Qwen exhausted eight turns inspecting evidence without submitting.
5. `bounded_v1` forced the submission tool after the required inspections. Qwen repeatedly returned 32 decisions where 99 were required; the environment rejected those incomplete answers.
6. `interval_v1` accepts an exact partition of the time grid into model-selected decisions. The environment expands those ranges only after validating coverage, bounds and labels. The controller does not pick a candidate, infer missing answers or read ground truth.

Do not pool these attempts into a model-performance average. Use the final interface's results for the smoke test and retain the earlier attempts as development/operational failures. If reporting tool-use robustness, disclose the controller and all attempts rather than presenting successful execution as unassisted behavior.

## Explanation audit

On the speaking clip, the final interval interface selected FastDVDnet. Qwen's rationale said it had the lowest disagreement with captured evidence among the restorations. That comparison is incorrect: FastDVDnet changes 11/99 decisions, whereas spatial NLM changes 9/99. The selected FastDVDnet predictions nevertheless have fewer errors than capture on this clip (62 versus 73). A useful output and an inaccurate explanation can coexist; the rationale is not independent evidence of correctness.

## Interpretation

This batch contains no restoration-induced correct-to-wrong transitions for either tested restorer. FastDVDnet corrects 11 timestamps and spatial NLM corrects nine, all on the speaking clip. The other two clips have unchanged detector predictions. This batch supports testing restoration benefits and validates the data/tool workflow; it does not provide a new positive example of restoration-induced attribution harm. Do not substitute this result for prior experiments or claim the research hypothesis is proved or disproved by one source.

FastDVDnet is video-only. The agent receives measured audiovisual detector evidence through tools, not native audio/video inputs. No audio-conditioned hallucination claim, learned risk assessor, validated safeguard, or multi-dataset generalization result is established here.

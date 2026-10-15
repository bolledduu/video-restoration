# Latest continuation: completed Qwen agent and learned-temporal pilot (2026-10-15)

See [QWEN_AGENT_PILOT.md](QWEN_AGENT_PILOT.md) and [QWEN_AGENT_TRACES.md](QWEN_AGENT_TRACES.md). All local inference and all eight actual Groq Qwen tool-agent cases finished. No jobs remain running. No credentials are stored in project artifacts.

The independent-source primary screen selected two previously untested UniTalk sources (kgVz_AXYvn0 and EEey6Pt0rqI), balanced speaking/silent, with sigma 15/40 noise. Forty candidate/reference evaluations used official learned five-frame FastDVDnet clipped-noise weights, spatial NLM, TalkNet, BRISQUE and MUSIQ on CPU. A separate AMI ES2002a meeting check added six evaluations on two transcript-proxy-labelled segments, reported separately. Forty-six local candidate/reference evaluations total.

The fixed primary screen FAILED: agreed harmful quality selection appeared on one of two sources, not the required two; useful restoration appeared on one source. On EEey6Pt0rqI silent sigma 15, both IQA metrics select spatial NLM and speaker error rises 0/24 to 2/24. At sigma 40, MUSIQ selects FastDVDnet; SSIM rises 0.117 to 0.927 while errors rise 0/24 to 5/24. BRISQUE selects a different candidate in this learned-restoration case. These new wrong scores are modest (about 0.51–0.60), not highly confident attribution. A different speaking clip is already all-wrong in clean reference, so do not explain every restoration regression as loss of genuine evidence. AMI shows no restoration-induced attribution change and a clean-reference speaking miss.

Mean primary frame error: captured 18.23%, BRISQUE 30.21%, MUSIQ 16.67%, confidence 25.52%, fixed disagreement guard 19.27%, oracle 13.02%. Qwen/qwen3.8-27b was account-verified and ran 8/8 actual cases with label-blind tools over cached model outputs. It scored 7/8 majority speaking answers and 18.23% selected frame error, matching captured. It does not directly see/hear media. Its first explanation miscounts 8 positive frames as 3; offline arithmetic is authoritative. Full API traces and usage are saved.

Audit passed protocol/code/input/output hashes, source disjointness, source archives, weight hashes, frame arithmetic, audio invariance, selectors, guard and agent label isolation. This is a completed bounded pilot with a failed stronger screen, not a completed novel assessor or paper. Existing RQs remain the scope; no novel-method or full-paper GO claim is warranted. No OCR pivot, paid GPU, human perceptual study, or audio-conditioned restoration occurred in this run.

---

# Earlier continuation: three-source temporal-denoising transfer probe

The user requested a fast 1–3-sample confirmation attempt with limited remaining usage. See [QUICK_TEMPORAL_PILOT.md](QUICK_TEMPORAL_PILOT.md). Three previously unscored cached UniTalk sources were selected by a fixed eligibility/timestamp rule before inference. All three selected tracks were silent; no speaking-track or quotation-agent claim is supported.

CPU OpenCV spatial and five-frame temporal colored nonlocal means were tested on fixed-seed noisy captures. This is a conventional denoising transfer probe, not the proposed learned temporal/generative confirmation experiment. BRISQUE and MUSIQ selections both had 0% speaker error, as did retaining captured inputs. One temporal candidate introduced one false detection among 24 scored frames, but neither assessor selected it. Agreed harmful selections: 0/3; improved attribution: 0/3 (captured-input error was already zero). The predefined feasibility screen FAILED; this is no additional support for scaling, not a disproof of the prior generative-restoration finding.

All 12 reference/candidate evaluations completed, with 36 MUSIQ/BRISQUE sampled-frame evaluations each. Audio-only predictions were invariant. The independent audit passed hashes, source disjointness, probabilities-to-errors, selections, safeguards and screen arithmetic. Files: quick_temporal_protocol.json, scripts/quick_temporal_pilot.py, scripts/audit_quick_temporal.py, results/quick_temporal_pilot.json, results/quick_temporal_audit.json. No jobs remain running. No human-quality study, learned temporal restorer, or novel verifier was evaluated. Do not promote this limited probe to full confirmation or a full-paper GO.

---

# Latest continuation: cross-metric failure discovery

See [WVAQ_FAILURE_DISCOVERY.md](WVAQ_FAILURE_DISCOVERY.md) for results, prior-work comparison and the three candidate RQs.

**Decision: YES to planning independent validation of quality-assessor consensus failures.** This is a narrower retrospective discovery hypothesis, explicitly chosen because native 8x8 was the previously observed failure stratum. Protocol 90e11cc preceded MUSIQ inference. It does not overturn the earlier failed cross-resolution screen and does not establish paper readiness or high novelty.

321 MUSIQ CPU evaluations, including an identical repeat, completed. Both BRISQUE and MUSIQ rank the same five evenly spaced frames for each candidate; existing 24-frame speaker outcomes are reused with hash checks. At 8x8, degraded-only speaker error is 33.33%, BRISQUE-selected error 75.00%, and MUSIQ-selected error 80.21%. Both metrics agree on five clips; all five increase error versus captured input, spanning three sources. On those same five clips, error rises 12.50% to 69.17%. At 4x4, MUSIQ improves pooled error from 47.40% to 41.67%; retain this counterexample.

The separate arithmetic audit passed. All 16 clip-resolution cases are reported. No independent new-source validation, novel safeguard, or human quality evaluation occurred. The model is still a framewise restorer, and suitability for speaking motion remains a major alternative explanation.

The candidate direction is reliability of quality-assessor agreement for restoration selection, with input-conditioned risk assessment and evaluation against always-captured-input selection. LL-Bench, CDI, FGResQ, QualiTeacher and DISRQAD overlap substantially with broad quality/fidelity/ensemble claims. A large failure count does not establish novelty. Do not call this a first-of-its-kind contribution without deeper prior-work comparison and a validated method.

---

# Latest continuation: WVAQ quality-selection hypothesis

The user requires every research question to center directly on WVAQ quality assessment; see AGENTS.md. The completed new experiment is [WVAQ_INITIAL_HYPOTHESIS.md](WVAQ_INITIAL_HYPOTHESIS.md).

The fixed hypothesis tested whether selecting the best blind image-quality score from a larger enhancement menu increases labeled speaker errors. Protocol e9c1592 preceded new BRISQUE scoring. Existing eight labeled clips/four sources and restoration outputs were reused transparently; 48 new conventional-candidate speaker inferences and 2,400 BRISQUE frame evaluations ran on CPU. This is retrospective feasibility, not unseen-data confirmation.

**Decision: the fixed overall hypothesis FAILED.** Pooled error increased 50.07% to 59.11% while BRISQUE improved 57.652 to 42.612. Three of four sources showed increased error. However, error rose 23.44 percentage points at 8x8 and fell 5.34 points at 4x4, failing the required positive effect at both resolutions. Conventional sharpening showed no comparable pooled increase; duplicate controls were invariant. Always using degraded video achieved 40.36% error. A concrete assessor failure exists, but the tested generalization did not pass and full-paper work is not justified by this pilot alone.

All 28 software tests and the independent arithmetic audit passed. Complete per-frame scores, exact subset-selection curves, model hashes, and gate checks are saved in results/wvaq_quality_pilot.json. No GPU resources used. No new safeguard or end-to-end agent was validated. The repeated-evidence and hypothesis-conditioned-restoration ideas remain untested. Do not reinterpret this completed test as having evaluated them.

---

# Research handoff — completed labeled CPU pilot

**Current decision: REJECT scaling this tested setup under the original quality-plus-error hypothesis.** Read [the final report](FINAL_CPU_RESEARCH_DECISION.md) for the complete operational research questions, results and reproduction instructions.

The labeled CPU pilot is complete. Eight one-second UniTalk clips from four source videos, balanced speaking/non-speaking, were restored at two resolutions using the official Ear-for-Face-SR model applied framewise. Original and swapped restoration audio were compared while captured frames and downstream TalkNet audio stayed fixed. Visual-only restoration and downstream visual-only scoring were included. All 16 condition/degradation combinations and 1,216 retained forward passes finished.

## What the evidence establishes

- Audio intervention caused labeled speaker-decision errors: 25 correct-to-wrong frame decisions, alongside 45 wrong-to-correct decisions. Swapping audio did not uniformly worsen attribution.
- Zero frames satisfied the predefined conjunction of improved SSIM, correct reference/degraded/matched predictions, and a swapped-audio error with a wrong-class score increase of at least 0.10. RQ1 failed its screen.
- The stronger degradation produced fewer correct-to-wrong transitions (2 versus 23), so an increasing degradation-to-harm trend was not observed.
- Fixed safeguards passed the RQ3 screen. Captured-video disagreement ranking reduced retained error from 55.56% to 45.83% at matched 75% coverage versus confidence selection.
- The combined gate required both RQ1 and RQ3; it therefore failed. The criterion was committed before inference and was not relaxed after seeing results.

## Scope and verification

This is a small framewise-restoration pilot with dataset-author labels, not an independently relabeled population study. The restorer receives a fixed four-second audio context, not time-varying audio. It was designed for face/voice identity cues rather than learned temporal lip synchronization. Errors and softmax changes do not establish a specific hallucinated anatomical detail or calibrated confidence. The results do not disprove every audio-conditioned restoration architecture.

All cases, including reference-classifier errors, are retained. Selection preceded inference. Input alignment, output checksums, full-path deterministic repeats and downstream audio-only invariance passed. A separate standard-library audit verified saved-score arithmetic and decision logic. No experiment execution remains pending within this declared CPU pilot.

## Artifacts

The final report and `results/unitalk_*` hold the protocol-linked measurements, frame scores, label/source provenance and audits. `data/unitalk/all_case_comparison.png` shows all 16 cases locally; media and model weights are intentionally excluded from Git. Reproduction is documented in the final report.

## Historical context

Earlier AV-SpeakerBench annotations remain provisional and were not used as ground truth here. Earlier OmniVR author-output evaluations did not isolate conditioning audio; the prepared 42-job temporal GPU experiment remains unexecuted and is outside this result. LL-Bench was a separate side experiment and supplies no GO decision for these research questions. No GPU work or spending is scheduled.

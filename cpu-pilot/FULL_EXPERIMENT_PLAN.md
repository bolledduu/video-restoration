# Full experiment plan: preserving speaker evidence in restoration-assisted agents

**Design date: 15 October 2026. Status: proposed protocol, not executed or preregistered.**

The completed pilot motivates research; it does not establish the proposed method or the prevalence of the failure. Its stronger replication screen remains failed. No experiment below is designed to force a positive answer. Freeze the protocol, splits, model versions, thresholds and primary comparisons before inspecting new test outcomes.

## 1. The contribution we need to test

An audio-visual agent can use restoration to obtain a clearer view of a person. The research problem is deciding when that output remains reliable evidence about who spoke. The intended contribution is a task-aware audiovisual quality assessor and selection rule, supported by a multi-model evaluation. Adding an LLM controller, renaming prediction disagreement, or combining familiar scores is not sufficient novelty.

Three separate claims require evidence:

1. **RQ1 — failure characterization:** identify conditions in which quality-based selection, including assessor agreement, introduces speaker errors. Agreement need not be more harmful than disagreement; test that comparison rather than assuming it.
2. **RQ2 — assessment method:** estimate the risk that a candidate introduces errors, while distinguishing that risk from beneficial prediction changes. Show information beyond quality, confidence, audiovisual synchrony and input consistency.
3. **RQ3 — useful selection:** retain quality improvements and useful speaker corrections while reducing new errors, including errors in agents' person–statement assignments.

The paper can claim an observed failure without establishing the proposed lip-flicker mechanism. The mechanism requires separate controls. Do not claim that commercial assistants enhance mouths, authorize purchases using this pipeline, or send incorrect summaries unless directly documented and tested. Applications motivate the task; they are not experimental observations.

## 2. What the supplied document actually specifies

The document names **Qwen-27B**, **Whisper**, **bicubic**, **sharpening**, and unspecified **neural enhancement**. It does not identify several agent checkpoints or a complete restoration roster. Qwen-27B is not itself a reproducible model identifier. The pilot used `qwen/qwen3.8-27b`; pin the serving model/version and date for a new run.

The named datasets are AVA-ActiveSpeaker, AMI, VoxConverse, DMD, Drive&Act, AVICAR, EasyCom, and CHiME-6/OpenSLR150. The following audit determines their roles; the document's scenario descriptions are not accepted as dataset facts.

| Dataset | What the official material supports | Planned role | Required check / limitation |
|---|---|---|---|
| AVA-ActiveSpeaker | Face tracks with audible-speaking, inaudible-speaking and non-speaking labels | Core controlled speaker-evidence benchmark | Preserve all original labels; report inaudible speech separately. Do not infer transcripts or named-person labels. Use actual annotation timestamps, not a universal assumed frame rate. |
| AMI | Synchronized audiovisual recordings, transcripts and participant/camera mappings | Core multi-person attribution and task transfer | Verify media synchronization and camera/participant mapping. Transcript segments are not automatically dense visual ASD ground truth. Split related meetings by recording group and participants. |
| EasyCom | Egocentric video, microphone audio, voice activity, transcripts, face/head boxes and participant IDs | Core held-out conversational-domain test | Restaurant-like conversation, not a validated home-payment benchmark. Respect off-camera participants and annotation defects. Inspect participant reuse between sessions. |
| VoxConverse | Multispeaker diarization annotations and audiovisual source provenance | Optional external test | RTTM speaker clusters alone do not identify face tracks. Confirm video availability and manually verify speaker-to-visible-person mapping. Official site currently reports audio unavailable there. |
| AVICAR | In-car speech corpus with eight audio and four video channels | Optional noise/domain extension after access validation | Read speech by speakers does not establish simultaneous driver-versus-passenger attribution. Check actual participants visible and available labels before assigning that task. |
| DMD | Driver behavior video/annotations; current repository describes a revised RGB real-car release | Exclude from core attribution benchmark | Do not assume synchronized public speech audio, word transcripts or driver/passenger speaker ground truth. Action labels are not speech labels. |
| Drive&Act | Multiview video, pose and activity annotations | Exclude from core attribution benchmark | Its documented modalities do not establish a usable audio/transcript/speaker-face attribution benchmark. |
| CHiME-6 / OpenSLR150 | Multichannel speech, transcripts and floorplans in the linked release | Optional audio-only robustness source | The linked release is not a synchronized audiovisual dataset for testing restored face evidence. Do not manufacture paired video by attaching unrelated footage. |

The current pilot used a UniTalk-ASD distribution with its published labels. Do not relabel that experiment as an official AVA evaluation. Any reuse or overlap with AVA must be established from source IDs and timestamps, then removed from held-out testing.

Sources: [AVA labels](https://sites.research.google/gr/ava/download/), [AMI corpus](https://groups.inf.ed.ac.uk/ami/corpus/), [AMI signal mapping](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml), [AMI known issues](https://groups.inf.ed.ac.uk/ami/corpus/dataproblems.shtml), [EasyCom](https://github.com/facebookresearch/EasyComDataset), [VoxConverse](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/), [AVICAR paper](https://people.csail.mit.edu/klivescu/ws06/papers/lee_etal_ICSLP04_AVICAR.pdf), [DMD current release](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset), [Drive&Act](https://www.driveandact.com/), [CHiME-6 release](https://openslr.org/150).

## 3. Model matrix: separate each role

The additions below are **proposed choices**, not models named in the document or already run in the pilot. Resolve downloads, licensing, compatibility and exact checkpoints before freezing the roster. Model count alone is not evidence of diversity.

| Role | Proposed core choices | Why included | Experimental restriction |
|---|---|---|---|
| Agent controller | Pilot Qwen checkpoint; GPT-OSS-20B; Llama-3.3-70B-Instruct | Three model families rather than several sizes of one family | Same text tool schemas, evidence, prompts and budgets. Provider/account availability must be verified; no silent substitutions. These are tool-mediated agents, not three native AV models. |
| Direct audio-video model | Optional Qwen3-Omni-30B-A3B-Instruct; a second family only after verifying native audio+video support | Tests whether failures extend beyond tool-mediated decisions | Report in a separate track; transcript-only or image-only access is not equivalent to native audiovisual input. Do not pool these results with the controller comparison. |
| Speech recognition | One fixed Whisper checkpoint, proposed large-v3 | Supplies the same transcript to all agent policies | Primary attribution test uses reference transcript with speaker IDs removed; a second test uses ASR output. Whisper does not supply visual person identity. |
| Speaker detection | TalkNet and Light-ASD; C3ASD as an additional contemporary robustness comparison if its official checkpoint runs | Test whether a failure or assessor merely exploits one detector | Calibrate thresholds on development data; no test-specific thresholds. Light-ASD and its derivatives are not fully independent architecture families. |
| Conventional restoration | Bicubic for low resolution; unsharp masking; spatial NLM for noise | Controls for resize, sharpening and conventional denoising | Bicubic is not a denoising control; preserve task-specific candidate sets. |
| Learned temporal restoration | FastDVDnet for noise; RealBasicVSR for low-resolution/compressed-video conditions | Include models designed for video, not only framewise face generators | Check intended input domain and checkpoint; no claim from mismatched noise levels or repeated single images. |
| Learned face restoration | CodeFormer as a framewise restoration family | Test a different restoration prior | Label it framewise; do not use it as the only evidence about temporal video restoration. |
| Quality assessment | BRISQUE, MUSIQ; DOVER technical and overall outputs reported distinctly | Classical image, learned image and temporal video perspectives | Identical content/frame sampling within each metric; a consensus is not automatically independent evidence. |
| Synchrony / fidelity baselines | SyncNet; input consistency; PSNR/SSIM/LPIPS against a reference where available | Compare competing explanations | SyncNet score is not ground truth. Reference metrics are evaluation-only, never deployable selector inputs. |

Official implementation references: [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD), [Light-ASD](https://github.com/Junhua-Liao/Light-ASD), [C3ASD](https://github.com/jisoo-o/C3ASD), [FastDVDnet](https://github.com/m-tassano/fastdvdnet), [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR), [CodeFormer](https://github.com/sczhou/CodeFormer), [DOVER](https://github.com/VQAssessment/DOVER), [SyncNet](https://github.com/joonson/syncnet_python), [Whisper](https://github.com/openai/whisper), [Groq model availability](https://console.groq.com/docs/models), [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni).

The method comparison also needs applicable current restoration-fidelity assessors from the existing literature audit, including CDI/HS/SHAFE where code, model and input requirements permit. A method requiring a clean reference belongs in an oracle/reference-only panel. Confirm the exact implementation and paper before presenting any as reproduced. An unavailable baseline must be documented, not replaced with an invented implementation carrying its name.

## 4. Data units, labels and splits

### Units

- **ASD unit:** a visible face track with time-aligned audible-speaking labels. Report official benchmark scoring separately from windowed diagnostic scoring.
- **Attribution unit:** a real utterance with a verified speaker-to-person mapping and at least two visible candidate people for the primary wrong-person test. Preserve offscreen and overlap cases as separate strata.
- **Statistical unit:** original source/meeting group, not a frame, face crop, noise realization, camera view or repeated agent response.
- **Agent episode:** one fixed event and candidate set, ending in a structured person–statement record. Use anonymous person IDs that remain stable within the episode.

Use three-second target intervals with one second of surrounding context on each side as a starting offline protocol. Do not convert a whole three-second interval to a single speaker label if it contains transitions. Score time-aligned activity; use utterance boundaries for attribution. Run a preselected 1/3/5-second target-context sensitivity test, with the same events where possible. Preserve time stamps and audio alignment when resampling.

Primary labels: audible speaker, silent listener, speaking-but-inaudible, offscreen speaker, overlap, and uncertain/unscorable. Do not collapse offscreen into a visible person's identity. Uncertain annotations are adjudicated or excluded by a frozen rule; model uncertainty is an output and cannot be used to remove a test example.

For attribution, obtain two independent annotations of speaker-to-face mapping for the manually labeled subset, then adjudicate disagreements. Track missing/cropped faces, mouth occlusion, laughter, chewing, nodding, face size and head pose. Report class-specific agreement and adjudication counts. Do not infer chewing from a speaker-detector error. Audit a random sample of inherited labels as well as disputed cases.

### Splits and planning scale

Use official train/validation boundaries where appropriate, then split the method-development pool by original source. Separate training, tuning, calibration and locked testing. All overlapping windows, multiple cameras and degraded/restored variants stay in the same split. AMI submeetings sharing participants must not be divided across train/test. For EasyCom, verify participant/session overlap; treating the entire dataset as an untouched external test is preferable to a random clip split.

**Planning targets, not claims of sufficient power:** approximately 600 AVA test windows from at least 20 available test sources, 300 AMI windows across about 10 eligible held-out recording groups, and 300 EasyCom windows across all eligible independent held-out groups. Report actual numbers and reduce claims if access or grouping yields fewer independent units. Reserve all previously inspected pilot sources for development, including both UniTalk pilot cohorts. Do not reach a sample target by adding neighboring frames from the same recording.

These 1,200 windows are a resource estimate. Before locking the test, simulate source-clustered paired differences using a separate development sample, targeting a prespecified practically meaningful effect (proposed 2 percentage-point attribution-error reduction) and 80% power at a two-sided 5% level. If independent sources are insufficient, report a precision-limited study rather than treating 1,200 windows as sufficient by definition. Class-balanced diagnostic samples and natural-prevalence samples must be reported separately.

## 5. Controlled conditions and causal checks

For each held-out captured event, create paired variants. Keep original audio, transcript, target interval, face identity, target boxes, detector weights and question fixed during the primary visual intervention. Save clean-source predictions too: a clean-source miss must not be described as restoration inventing a mistake from otherwise reliable evidence.

Initial degradation grid, frozen after development-only inspection:

| Family | Mild / severe proposal | Matched restoration | What it tests |
|---|---|---|---|
| Noise | Gaussian sigma 15 / 40 on 8-bit RGB | NLM, FastDVDnet, eligible face restoration | Continuation of the controlled pilot; separately add realistic sensor noise only with a specified model |
| Low resolution | Spatial downsampling 2x / 4x, documented kernel | Bicubic, RealBasicVSR, eligible face restoration | Loss of mouth detail; report native face sizes, not only resized network inputs |
| Compression | H.264 CRF 28 / 40 with fixed encoder settings | Checkpoint-supported video restoration plus conventional controls | Block/ringing artifacts; CRF is not comparable across codecs without reporting bitrate |
| Blur | Linear motion kernel 5 / 11 pixels at declared evaluation resolution | Appropriately matched restoration checkpoint | Motion degradation; don't call Gaussian blur camera motion |
| Clean/no added degradation | Original source | Identity and all applicable candidates | Detect unnecessary processing and baseline detector errors |

The eight single-degradation conditions plus clean control are the core grid. Natural low-quality captures are a separate test without clean-reference metrics. A small, predeclared combined-degradation test follows; do not run every Cartesian combination. Frame freezes, audio/video offsets, low light and real network-loss traces are extensions with separate labels and time-handling checks. Gaussian noise is not a Wi-Fi simulation.

Controls required for explanations:

1. Identity/duplicate-output control: identical arrays must yield identical deterministic detector outputs; agent stochasticity is quantified separately.
2. Clean-reference control: report results for all cases and the captured/reference-correct subset. Never silently filter the primary result by clean-model correctness.
3. Audio-only and video-only detector controls, with captured-audio hashes fixed across visual variants.
4. Fixed ground-truth tracks in the controlled experiment; add detected tracks in a separate deployment test so tracking errors are visible.
5. Common output size/time grid and fixed downstream preprocessing; evaluate original restoration pixels before lossy re-encoding. Include a native-output-resolution sensitivity test.
6. For the mouth-artifact explanation, compare mouth-region and equally sized non-mouth-region substitutions, use the same blending procedure, include sham substitutions, and inspect detector changes. This is a post-restoration intervention, not proof by itself of physical mouth-motion hallucination.
7. If claiming temporal flicker, quantify motion-compensated temporal differences on valid visible mouth regions and have humans inspect representative clips. Naive temporal pixel differences confound real head/mouth motion.
8. If claiming audio-conditioned restoration causes false evidence, include a genuinely audio-conditioned model and matched/silenced/swapped conditioning audio while holding downstream audio fixed. The current visual-restoration roster cannot support that claim.

## 6. Agent environment and fair comparison

The core agent receives the same task and measurements for every policy. It cannot read ground truth, clean references, reference fidelity scores, evaluator error rates or oracle selections. No example filenames or identities encode the answer.

Primary question: **“Who said this utterance during [start, end]? Choose one visible person ID, offscreen, multiple speakers, or abstain, and cite the time interval and evidence used.”** The primary single-speaker subset has one adjudicated answer. Overlap uses set-valued scoring and is reported separately. A given utterance with its speaker tag stripped isolates attribution from ASR. Include transcript-only and audio-only controls to expose identity cues or answer leakage.

Tools: inspect captured tracks; request an eligible restoration; obtain IQA/VQA scores; obtain audiovisual speaker scores; compare captured/restored evidence; query the frozen assessor; submit the attribution record. Add a simple sandbox evidence ledger that records person ID, utterance ID and timestamp. This tests an actual structured output affected by attribution, without sending real emails or executing purchases. Free-form meeting summaries are optional, not the primary endpoint.

There are two comparisons:

- **Fixed evidence set:** all eligible outputs are cached and each policy can access the same measurements. This isolates selection and reasoning. Include deterministic selectors that require no language model.
- **Budgeted tool use:** agents choose tools within the same call, context and inference budgets. Charge for restoration computation even when outputs are served from cache for repeatability. Report model-specific tokenization and effective compute; equal call counts alone are insufficient.

Proposed core budget: 8 controller turns, 12 external tool invocations, 5 candidate outputs including captured, 2,048 generated tokens per turn, fixed maximum context. Inspect requests for untrusted tool content but give every controller the same schema. Use three prespecified decoding seeds/repeats. Temperature zero does not guarantee deterministic hosted inference. Record invalid output, tool failure and budget exhaustion; retain them in the denominator and report them separately from wrong-person attribution.

The main study is offline. A continuous-stream claim additionally needs sequential, causal replay: no future frames, fixed lookback, explicit decision delay, state reset policy, and measured p50/p95 latency. An offline model seeing future context cannot be presented as an online agent.

## 7. Candidate method: make the research testable without claiming novelty early

**Working design: a verifier of speaker evidence supported by the captured input. This is a hypothesis, not an established novel method.**

Inputs are the captured track D, restored candidate R and unchanged audio A. Estimate (i) the risk of a new error and (ii) the potential for correcting an existing error. A single unsigned disagreement score cannot express both. A proposed implementation compares aligned local mouth/face temporal features, audio–mouth consistency before/after processing, and the visibility of the relevant region in D. Unsupported detail should carry more uncertainty when D is severely degraded or occluded. Features outside the mouth and standard consistency scores remain explicit controls.

Train on source-disjoint paired data with published/verified activity labels. For a named downstream detector, the harmful-change target is 1 when its captured decision is correct and its restored decision is wrong; correction is defined conversely. This target predicts detector behavior, not physical truth about invented pixels. Hold out a detector at test time to support a broader evidence-preservation claim. Human activity-preservation judgments are a separate target/evaluation if available.

Selection maximizes a development-normalized quality benefit subject to an estimated harm budget, with captured input always present. Calibrate the tradeoff on an independent calibration split. Proposed operating point: at most 1 new error per 100 eligible events, with 0.5% and 2% sensitivity points; these are research settings, not deployment safety thresholds. Abstention is a separate extension, not an easy way to win the full-coverage comparison.

Before training, compare this mechanism with the existing fidelity assessors, synchrony methods and contemporary robust ASD work such as C3ASD. If it amounts to a known score fusion, treat it as a baseline. A method claim needs an explicit technical difference and ablations showing why it matters. Theory or a risk guarantee must not be claimed merely because a threshold was calibrated.

## 8. Metrics and exact denominators

Let C mean a correct decision and I an incorrect decision relative to verified ground truth. Compare captured and selected evidence on the same N eligible units. Compute this separately for frame-level ASD and utterance-level attribution. Do not mix the two endpoints.

| Metric | Definition / unit | Purpose |
|---|---|---|
| New-error rate H | count(C→I) / N | Absolute harm introduced by selection |
| Correction rate G | count(I→C) / N | Useful improvements retained |
| Net accuracy gain | G − H, in percentage points | Exactly equals selected minus captured accuracy at full coverage on the same units |
| Conditional regression | count(C→I) / count(captured C) | How often initially correct evidence is damaged; report denominator |
| Conditional correction | count(I→C) / count(captured I) | How often initial mistakes are repaired; report denominator |
| False-speaking rate | false positives / ground-truth non-speaking frames | Silent-person failure; separate speaking-but-inaudible labels |
| Missed-speaking rate | false negatives / audible-speaking frames | Cost of becoming overly conservative |
| ASD mAP | Official benchmark evaluator and label mapping | Comparable established benchmark result; not interchangeable with accuracy |
| Person–utterance accuracy | Correct person assignments / all eligible utterances | Main agent application endpoint; invalid/abstaining outputs are not silently removed |
| Wrong-person rate | Wrong visible person assignments / all eligible utterances | Directly captures assigning Alice's words to Bob; separate offscreen/overlap/invalid cases |
| Harm-detector AP (AUPRC) | Average precision on harmful-candidate labels; report prevalence | Primary ranking metric for rare harm; define implementation rather than mixing AP with trapezoidal PR area |
| Harm-detector AUROC | Area under ROC | Secondary ranking metric, not sufficient for rare harms |
| Recall at fixed FPR | Harm recall at 5% FPR; threshold from calibration | Detect failures without rejecting everything; report achieved test FPR |
| Brier score / reliability plot | Probability error against the declared harm or correctness target | Calibration; self-reported LLM confidence is not calibrated by default |
| Coverage | Non-abstained eligible answers / all eligible questions | Makes refusal visible; offscreen is a valid class, not abstention |
| Selective risk / AURC | Error among answered questions across coverage levels | Compare abstaining methods at the same coverage and report actual coverage |
| Human quality preference | Win/tie/loss against captured, randomized/blinded | Evidence of perceived visual improvement |
| MUSIQ / BRISQUE / DOVER | Individual metric changes, never one averaged raw number | Objective quality diagnostics; distinguish technical/aesthetic outputs |
| SSIM / PSNR / LPIPS | Against the undegraded source only in paired synthetic data | Reference similarity, not human perceptual preference or speaker truth |
| WER | (substitutions + deletions + insertions) / reference words | ASR diagnostic; with unchanged audio and cached Whisper, WER is invariant across visual policies |
| Cost and latency | Calls, tokens, wall time, p50/p95 per episode; hardware/API specified | Shows whether gains require more computation |

For continuous diarization only, add DER with a declared collar, overlap scoring and official implementation. DER speaker-cluster matching does not prove correct visual-person identity; keep person–utterance accuracy. Speaker-attributed WER may be a supplement once its identity mapping and transcription protocol are specified; ordinary WER cannot show who said the words.

### Assessor agreement

Define agreement before testing: BRISQUE and MUSIQ choose the same top candidate from the same eligible set with deterministic tie handling. Report agreement including captured, agreement on a restored candidate, the support count, and harm conditional on each. For three assessors, unanimous agreement is distinct from a majority vote. Also report harm when assessors disagree. Compare within degradation/quality/source strata; an aggregate difference alone is not proof that agreement causes harm. Shared training data or backbones prevent treating assessors as statistically independent witnesses.

## 9. Baselines reviewers need

1. Always captured; identity-copy control.
2. Each fixed restorer, including the best single restorer selected on development data, not test labels.
3. Best BRISQUE, best MUSIQ and best DOVER; separate score directions and content sampling.
4. Development-normalized rank ensemble and conservative ensemble; consensus fallback to captured.
5. Highest calibrated ASD confidence; captured/restored disagreement; input pixel/perceptual consistency.
6. SyncNet-based comparison and applicable published input-aware fidelity assessor implementations.
7. A simple learned logistic or boosted-tree fusion with the same inputs, labels and tuning budget as the proposed method. This tests whether architecture is necessary.
8. A calibration-based risk-control selector under its stated assumptions. Include conformal risk control only with a valid bounded loss, suitable selection policy and correct grouping/exchangeability assumptions; do not attach an unjustified guarantee under dataset shift. [Reference](https://arxiv.org/abs/2208.02814).
9. Agent without the proposed tool; agent with only a prompt warning; agent with disagreement; agent with the proposed assessor. Separate tool access from prompt changes.
10. Label-based oracle selection as a diagnostic ceiling, clearly marked non-deployable. Specify its tie rule and quality objective; do not present it as a baseline available at inference.

All trainable baselines receive identical label budgets and source splits. Report extra pretrained components and training/selection compute. A direct audiovisual model and a text-only tool controller do not receive identical inputs; compare them in separate panels.

## 10. Paper tables: exact designs and why they exist

Every reported rate includes its denominator in the table or caption. Main differences have paired, source-clustered 95% confidence intervals. Use a few readable panels rather than one table with 25 columns. Populate all cells from saved evaluation files; use “not applicable” or “unavailable”, never zero for missing runs.

### Table 1 — Data and annotation validity (main text)

**Rows:** dataset × split, with synthetic and natural capture tests separate.

| Dataset / split | Independent groups | Participants | Utterances / tracks | Scored duration | Silent / overlap share | Person mapping source | Manual audit / exclusions |
|---|---:|---:|---:|---:|---:|---|---|
| Planned; fill from frozen manifest | — | — | — | — | — | — | — |

**Why:** demonstrates that source counts, labels and person identities actually support the task. Define the grouping unit for each dataset in the caption. Include a supplementary source-flow table: available → accessible → eligible → excluded, with reasons.

### Table 2 — Does restoration preserve speaker evidence? (main text, RQ1)

**Rows:** degradation family/severity × restorer; separate dataset/detector panels.

| Degradation / restorer | N groups / units | Captured error ↓ | Restored error ↓ | C→I / N ↓ | I→C / N ↑ | ΔMUSIQ ↑ | ΔDOVER ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Planned | — | — | — | — | — | — | — |

**Why:** prevents reporting only regressions while hiding benefits or baseline detector weakness. Caption distinguishes reference-correct diagnostic strata from the complete population. SSIM/LPIPS and per-label false/missed speaking rates are in the supplement; the main human-quality comparison is Table 4B.

### Table 3 — Can assessor agreement be trusted? (main or compact supplement, RQ1)

| Agreement group | Supporting N / total | Restored selection share | C→I / N ↓ | I→C / N ↑ | Conditional regression ↓ | Δ error vs captured [CI] |
|---|---:|---:|---:|---:|---:|---:|
| Same restored winner / same captured winner / disagreement | — | — | — | — | — | — |

**Why:** tests the actual agreement question, not merely whether a metric sometimes fails. Report raw counts and stratified or adjusted contrasts. Same-captured harm is zero by construction; it is not empirical proof of a detector's correctness.

### Table 4A — Can the assessor detect harmful candidates? (main text, RQ2)

| Assessor | Inputs / supervision | AP ↑ | AUROC ↑ | Recall@5% FPR ↑ | Achieved test FPR ↓ | Brier ↓ |
|---|---|---:|---:|---:|---:|---:|
| Quality / confidence / disagreement / synchrony / fidelity / learned fusion / proposed | — | — | — | — | — | — |

**Why:** checks whether the proposed method adds information beyond cheaper, existing signals. Caption reports harm prevalence, number of independent groups and baseline-matched training data. Candidate observations from the same source stay clustered.

### Table 4B — Does selection preserve useful quality? (main text, RQ3)

| Selector | Human win / tie / loss ↑ | Δ technical VQA ↑ | Human unacceptable-output rate ↓ | Restoration acceptance | p95 latency ↓ |
|---|---:|---:|---:|---:|---:|
| Captured / quality-only / strongest safe baseline / proposed | — | — | — | — | — |

**Why:** rejects the trivial solution of never restoring or returning an oversmoothed image. Human judgments, not the optimized metric alone, support a perceptual-quality claim. Predefine what “unacceptable” means in the rating instructions.

### Table 5 — Main full-coverage selection result (main text, RQ3)

| Selector | Person–utterance accuracy ↑ | Wrong-person rate ↓ | C→I / N ↓ | I→C / N ↑ | Net gain [CI] ↑ |
|---|---:|---:|---:|---:|---:|
| Captured / best fixed restorer / quality-only / strongest calibrated baseline / proposed / oracle | — | — | — | — | — |

**Why:** directly tests whether the method helps the agent's real task at full coverage. Dataset and detector results use separate panels. Abstention is disabled here or counted as an incorrect/non-completed answer consistently. Pair with Table 4B rather than hiding the quality tradeoff in a weighted scalar.

### Table 6 — Does the result transfer across agent models? (main text)

| Agent / input mode | Assessment policy | Attribution accuracy ↑ | New wrong-person errors / N ↓ | Invalid / no-answer rate ↓ | Δ accuracy [CI] ↑ | Tool calls / cost ↓ |
|---|---|---:|---:|---:|---:|---:|
| Each frozen controller family × captured / quality-only / existing guard / proposed | — | — | — | — | — | — |

**Why:** shows whether benefit survives a different controller and whether it only comes from extra tools. Compare each policy within an agent before cross-agent aggregation. Tool-controller and native AV tracks have separate panels. Each reported result averages the prespecified repeats, not the best run.

### Table 7 — Generalization and method ablations (main text or supplement)

Panel A: **Held-out axis | Test group count | Best baseline error | Proposed error | Δ error [CI] | New-error rate | Δ quality.** Axes: source, dataset, restoration family, detector; use a source-disjoint test for every axis. Keep joint multi-axis holdout separate from one-axis holdouts.

Panel B: **Method variant | AP | Attribution error | C→I/N | I→C/N | Quality preference | Extra cost.** Rows: full; no captured input; no temporal features; no audio comparison; no visibility estimate; harm-only vs harm+benefit; no calibration; global vs mouth-local analysis; equal-capacity simple fusion. These rows apply only to components actually implemented.

**Why:** tests whether the contribution survives new inputs and whether its proposed mechanism matters. It is not enough to tune on a restorer and test a different checkpoint from the same family.

### Supplementary tables

- **S1 model manifest:** role, exact checkpoint/commit, modality access, pretraining sources known/unknown, resolution/context, parameters, license, hardware, measured latency.
- **S2 corruption manifest:** codec/kernel/noise parameters, stage of application, native face sizes, bitrate, timestamps and seed.
- **S3 selective prediction:** selector, target coverage, actual coverage, answered-set error, wrong-person rate, quality, AURC. Show 80/90/95/100% coverage and complete curves.
- **S4 mechanism controls:** intervention, evaluated groups/units, clean/captured error, post-intervention error, temporal artifact score, human evidence-preservation judgment.
- **S5 transcription:** reference-transcript attribution, Whisper-transcript attribution, WER, mapping errors and transcript-only baseline. Do not claim a video selector improves unchanged-audio WER.
- **S6 failures/cost:** detector failures, tracker failures, bad alignment, model refusals, invalid JSON, tool errors, OOM/timeouts, episode cost and execution success. No silent denominator reduction.
- **S7 full per-dataset/per-model values:** all conditions and repeats, including counterexamples and negative results.

Main paper priority is Tables 1, 2, 4A, 4B/5 as a paired result, and 6. A short workshop paper should move the remaining tables to the supplement rather than compressing text until it is unreadable.

## 11. Figures

1. **System and causal diagram:** captured audio/video → restoration candidates → quality and evidence assessment → selected evidence → person–statement record. Mark unchanged audio, hidden labels and the three evaluation boundaries.
2. **Quality versus harm:** x = human quality preference or independently evaluated technical quality; y = new-error rate. Show captured, fixed restorers and selectors with source-clustered uncertainty. Display every tested operating point; do not draw a favorable Pareto frontier using test labels to choose the operating threshold.
3. **C→C / C→I / I→C / I→I transitions:** show harm and correction together, normalized by the same population, with counts. Different wrong-person identities inside I→I can be reported separately.
4. **Degradation/restoration map:** source-level harm across noise, blur, compression and resolution; show missing/inapplicable cells, not zeros.
5. **Risk–coverage curves:** matched coverage, common test set, explicit tie rule; accompany with full-coverage results.
6. **Inspectable case strips:** correct-to-wrong, wrong-to-correct and no-change examples, including an inherited clean-detector failure. Show time-aligned frames, audio timing, detector scores, candidate choices and final agent record. Select examples by a fixed stratified rule; keep them illustrative, not prevalence evidence.

## 12. Human quality and evidence study

Plan a stratified random subset of about 200 source-event/candidate pairs with five independent judgments each as an initial resource estimate, then adjust using development reliability estimates. This is not an automatic guarantee of statistical power. Balance datasets, conditions and methods; do not pick only successful outputs. Keep a separate failure-enriched diagnostic panel and never use it to estimate natural prevalence.

Use blinded, randomized left/right videos at matched size and playback rate with a tie option. Ask visual-quality preference without revealing detector errors or the proposed method. In a separate task, show synchronized audio/video and ask about speaking evidence; do not merge those answers into a single quality score. Source reference may be shown for the evidence task when available, and that condition must be identified. Record display conditions, rater eligibility, repeated checks, prespecified exclusions, inter-rater agreement and uncertainty clustered by source and rater. A VLM judge is an additional model, not an independent human study.

Use [ITU-T P.910](https://www.itu.int/rec/T-REC-P.910-202310-S) as a design reference; do not claim standards compliance unless the actual procedure meets its requirements.

## 13. Statistics, operating points and claims

- Primary method comparison: proposed versus the strongest eligible baseline chosen on development data, at full coverage, on person–utterance attribution. Show paired difference in accuracy/new-error rate and independent quality evidence.
- Use source-/meeting-group bootstrap intervals (proposed 10,000 resamples) for paired differences; retain all linked variants together. For very few independent groups, state the limitation and show individual group results rather than relying on narrow asymptotic intervals.
- Report both dataset-level macro results and transparent pooled counts. Do not let one large dataset or long track dominate silently. For label-balanced samples, identify their artificial prevalence.
- Use source-clustered power/precision planning and a fixed stop rule. Do not keep acquiring samples until a favorable p-value appears. A separate development expansion may guide a new prospectively locked test.
- Preselect primary comparisons and correct families of secondary significance tests, for example using Holm's procedure. Effect sizes and uncertainty matter more than stars on a table.
- Proposed meaningful improvement target: 2 percentage points in person–utterance accuracy versus the strongest baseline. Proposed quality criterion: human preference better than chance/tie-adjusted null versus captured, or a predefined non-inferiority margin versus the best quality selector. Set margins using development evidence and task requirements; they are not universal conference standards.
- For harm-budget selection, freeze thresholds on calibration data and report achieved test harm and quality, including violations. An empirical calibration target is not a guarantee under a new dataset or restorer.
- If native AV models or streamed agents are not actually evaluated, restrict claims to the measured tool-mediated offline setting.
- The prior eight-case pilot is discovery evidence. Do not count it as an independent confirmation sample, fabricate standard errors from its 192 frames, or revise its failed gate into a pass.

## 14. Execution plan and compute control

1. **Freeze the task and audit inputs:** source IDs, transcript/face mappings, label definitions, licenses, split manifests, successful baseline inference and input timing. Deliver Table 1 and exclusion flow before expensive generation.
2. **Development study:** approximately 100–150 windows from separate development groups. Validate matched restoration domains, human rating instructions, metric directions, baseline competence, and clustered power assumptions. Fix model roster and operating points afterward; do not inspect locked test outputs.
3. **Build reusable candidate cache:** clean/captured and applicable restored videos; compute both ASD backbones and all assessment signals once. Cache by input/model/parameter hashes. Retain failures and exact timings.
4. **Characterize failure:** populate Tables 2–3 without fitting to test outcomes. Keep harmful, helpful and unchanged strata.
5. **Develop and calibrate the assessor:** train on development/training sources only; compare simple fusion and published approaches before investing in elaborate architecture.
6. **Locked selector evaluation:** populate Tables 4–5 and generalization panels. Run human evaluation on a blinded, prespecified subset.
7. **Multi-model agent evaluation:** a source-stratified attribution subset, proposed 300 events with one prespecified degradation condition per event; four policies × three controllers × three repeats = **10,800 episodes**. This is a planning count, not a promise that it fits the existing cloud credit. Measure 20 development episodes to estimate actual time/tokens before allocation. Reduce design before locking if necessary, not after seeing results. Deterministic selectors are evaluated cheaply over the full cache.
8. **Optional extensions:** native audiovisual model track; causal streaming replay; VoxConverse/AVICAR transfer after their label/access checks. These are not prerequisites for the narrower core claim and should not delay it indefinitely.
9. **Independent recomputation and release:** recompute every table from raw logs; validate paired identities, input hashes, denominators and schema. Release code, configuration, manifests, permitted outputs, table-generation code and known limitations. Never publish credentials or redistribute restricted media without permission.

A rough upper bound for 1,200 windows × 9 conditions × 5 candidates is 54,000 candidate clips; the actual task-compatible matrix is smaller. Two detectors could require about 108,000 downstream evaluations. Do not silently create this full Cartesian workload on the laptop. Use the compatibility matrix, cache reuse, explicit budget estimates, and independent-source coverage as the priority over redundant variants.

## 15. Required run record

Each record must contain: dataset/release; original source/group/person IDs; split; native timestamps/fps/resolution and target interval; label provenance/uncertainty; face mapping/tracking status; degradation parameters/seed; candidate model/checkpoint/code hash; conditioning audio and downstream audio hashes; quality scores and sampling frames; detector/frame scores; clean/captured/candidate correctness (offline only); selector score and threshold version; agent model/version/input mode/prompt/schema; tool trace; selected evidence; final person–utterance answer; confidence type; parsing/timeout status; tokens/calls/wall time; annotation/human-study link; and exclusion reason if any.

Make table scripts assert: equal denominators for paired policies; G−H equals accuracy change at full coverage; no source leakage; no reference/label data in tool responses; audio invariance in visual interventions; and no excluded failed episodes hidden from end-to-end reporting.

## 16. What would support submission

Proceed toward a method paper when the new assessment mechanism is clearly different from applicable prior work, outperforms strong fair baselines on independent sources, retains measurable quality benefits, and its advantage survives a held-out restorer/detector or dataset consistent with the stated scope. Agent-level evidence should show correct person–statement assignments, not only a controller repeating speaker-score arrays.

If the method fails but the benchmark and failure analysis are substantial and well validated, evaluate a measurement/benchmark contribution on its own merits and novelty. Do not keep the method claim by hiding negative comparisons. No number of models, tables or datasets can guarantee acceptance or an award; the design above makes the claims testable and the evidence inspectable.

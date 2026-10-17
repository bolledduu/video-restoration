# TGIF-QA branch review and dataset handoff

## Decision

Reuse the experimental infrastructure. Do not scale TGIF-QA as the core test of the current speaker-evidence paper. It is a separate visual-question-answering robustness study, not audiovisual speaker attribution. Keep its negative primary result if the experiment is discussed. No manuscript claims or experiment results were changed by this review.

Reviewed branch: `claude/tgifqa-restoration-attribution`, commit `376950acaccb42d02736bf65ae0c4095d524c667`, repository `bolledduu/agent-audio-visual`. The linked branch exists; the slightly different branch name in the chat was not used.

## Independently checked evidence

Read the implementation, protocol, saved samples, preparation manifest, response records, tests, and report. Recomputed the saved scoring rule without executing the branch's API runner. This is a record/code audit, not a new inference run or visual inspection of downloaded GIFs.

- 1,430 sampled questions; 1,413 complete questions from 1,304 prepared GIFs; all prepared GIFs have eight sampled frames.
- Zero GIF-ID overlap with the pilot. This does not rule out duplicate visual content under different GIF IDs or pretrained-model exposure.
- 7,637 saved responses, zero duplicate question/condition/repetition keys.
- All records use requested model `Qwen/Qwen3.8-27B:deepinfra`, with recorded served model `Qwen/Qwen3.8-27B`.
- Zero discrepancies when independently reproducing the code's answer scoring; zero inconsistencies between response frame-hash lists and the preparation manifest. Media bytes and remote execution cannot be independently verified from those records alone.
- The git snapshot includes the protocol and outcomes together. Its statement that the protocol was written beforehand is documented, but an independently timestamped preregistration is not established by this commit.

| Input condition | Correct / complete | Accuracy |
|---|---:|---:|
| Source frames | 914 / 1,413 | 64.69% |
| Degraded | 785 / 1,413 | 55.56% |
| Bicubic | 774 / 1,413 | 54.78% |
| Unsharp mask | 769 / 1,413 | 54.42% |
| Real-ESRGAN | 756 / 1,413 | 53.50% |

| Comparison | Correct → wrong | Wrong → correct | Recomputed exact McNemar p |
|---|---:|---:|---:|
| Real-ESRGAN vs bicubic, primary | 97 | 79 | 0.19991 |
| Real-ESRGAN vs degraded, secondary | 102 | 73 | 0.03399 |

The primary net change is −1.27 percentage points and is not statistically significant under the stated test. The secondary −2.05-point change is unadjusted and does not pass the report's five-comparison Bonferroni threshold of 0.01. One question per GIF also gives an inconclusive primary result: 87 harmful versus 70 helpful, p=0.2015. Repeated calls change correctness in 3/286 degraded and 6/286 Real-ESRGAN cases. These are correctness changes, not a count of every textual-answer change.

## Fit to the four questions

| Current question | What this branch contributes | What it cannot establish |
|---|---|---|
| Quality and attribution | Paired processing can change visual QA correctness | Speaker-attribution errors; improved perceptual quality |
| Evidence assessment | A reusable harmful/helpful transition calculation | A validated audiovisual risk assessor |
| Selective repair | Bicubic and sharpening controls | An implemented repair or useful matched-coverage selection |
| Agent reliability and transfer | Direct frame-based Qwen responses | A tool-using agent policy, audio–face reasoning, repaired-agent transfer |

The GIF input carries no synchronized soundtrack or speaking-state/person–utterance labels. The runner sends images and a question directly to the model; it does not execute an agent tool loop. Its recorded Qwen deployment must remain distinct from the Groq tool-only pilot even where the model-family names look similar.

## Corrections before reuse or scaling

1. **Narrow the causal claim.** Same-resolution Real-ESRGAN vs bicubic controls output dimensions; it does not isolate hallucinated content. Denoising, sharpening, texture, reconstruction error, and backend variability also differ. Call this a restoration-method contrast. A single flip does not establish its pixel-level mechanism.
2. **Measure quality.** Saved PSNR averages are 24.100 dB for bicubic, 23.891 for sharpening, and 23.356 for Real-ESRGAN. There is no reported perceptual-quality gain, human preference, or IQA/VQA evidence. Add paired quality measurements before claiming “better quality, worse answers.”
3. **Audit semantic scoring.** `normalize` at line 395 reduces an answer to its first non-article token and applies simple plural stripping. This is not full normalized exact match. Review harmful and helpful cases blind to condition. For example, `FRAMEQA41023` is scored harmful for “motorcycle” → “bike”; without viewing the source and annotation convention, that is a potential semantic ambiguity, not a confirmed factual hallucination. “Horse” → “dog” in `FRAMEQA16187` is a clearer candidate for visual review, not yet independently verified here.
4. **Resolve count labels and temporal sampling.** Do not assume label zero means one without checking the official encoding. Eight index-uniform GIF frames can omit repetitions and ignore variable frame durations (`gif_frames`, line 162). This design is unsuitable as a dense mouth-motion assessment pipeline.
5. **Strengthen analysis independence.** Several questions share a GIF; use GIF-clustered intervals for the complete analysis, alongside the already useful one-question-per-GIF sensitivity. Do not describe repeat noise as directionally unbiased solely from its aggregate rate.
6. **Make resumption configuration-aware.** `stage_prepare` skips existing GIF entries (line 248); shared frame paths are keyed by GIF/condition. `stage_query` skips existing `(qid, condition, rep)` keys (line 347). Changes to degradation, seed, model, prompt, or media can therefore mix stale and fresh artifacts on reuse. Include a protocol/configuration hash in output paths and validate it, model, prompt, and frame hashes before reusing records. No such mixing was found in the audited snapshot; this is a scaling risk.

## Dataset acquisition priorities

| Priority | Dataset | Download together | Correct endpoint |
|---|---|---|---|
| Core perception test | AVA-ActiveSpeaker | Original audiovisual source windows plus official speaking labels, boxes, entity IDs, timestamps, and source split | Visible-face audible-speaking classification |
| Core agent attribution test | AMI | Matching room/participant video, synchronized audio, transcripts, participant/camera mapping, and signal timing metadata | Real utterance → visible participant, plus separate ASD labels if adjudicated |
| External transfer | EasyCom | Matching session video, microphone-array audio, participant-linked voice activity, face boxes, transcripts, and metadata | Person-linked conversational speech in egocentric footage |
| Conditional extension | VoxConverse | Source audiovisual material and RTTM, plus independently verified face–speaker mapping | Diarization/attribution only after mapping validation |
| Separate optional study | TGIF-QA | GIFs, QA annotations, sampling manifest | Visual QA; not speaker evidence |

For AVA preserve `SPEAKING_AND_AUDIBLE`, `SPEAKING_BUT_NOT_AUDIBLE`, and `NOT_SPEAKING`. Do not relabel inaudible speech as a silent person. Check original source overlap with UniTalk and earlier pilots. A dataset split alone does not establish absence from a pretrained model's training set.

For AMI, transcript timing is not automatically a dense visual speaking label. Establish participant-to-camera mapping and offset/drift before labeling a person. Use reference transcripts with speaker names removed when testing agent attribution; keep identities in the evaluator, never the prompt.

For EasyCom, the official documentation records nonconsecutive/redacted segments, an incorrect participant banner in Session 8, a metadata audio-channel typo, and differing face/head frame-index origins. Use verified participant annotations, remove identity banners from model input, respect segment boundaries, and synchronize the chosen streams. Close-microphone audio may have a sample rate near 48008.6 Hz rather than exactly 48000; do not assume timestamps align after an unchecked conversion.

## Small acquisition/validation batch before full download

Start with 12 eligible windows per core dataset, covering at least four source/meeting/session groups where available. Use these 36 windows to validate decoding, alignment, identities, model input, and scoring—not to estimate population effects or certify the method. Select windows by a fixed seed within annotation-defined strata before running the models. Record unavailable media and use the next predetermined eligible item, without inspecting predictions.

Include audible speaking, visible listeners, speaker changes, and separate overlap/offscreen cases. For the primary wrong-person endpoint require at least two visible candidates and a verified person–utterance mapping. Keep valid hard cases such as occlusion or small faces as declared strata instead of excluding them because models fail. Eight frames may be used for a defined VLM-input budget, but preserve continuous synchronized video for the ASD and temporal-restoration pipelines.

Save a manifest with dataset/release, source and participant group, split, time interval, frame rate/PTS, audio sample rate, synchronization offset, face/person IDs, annotation provenance, inclusion/exclusion reason, and hashes. Deduplicate original sources across pilot/development/test. Freeze all full-test choices before seeing new outcomes. Include beneficial, harmful, and unchanged outcomes; do not increase degradation until the hypothesis passes.

No bulk media downloads or new paid inference were run during this review. This report specifies the acquisition order and validation requirements; it does not mark those datasets as downloaded.

## Message to teammate

I reviewed your TGIF-QA branch and reproduced the main counts. The paired conditions, bicubic control, separate GIF split, and repeat checks are useful. The primary result is 97 harmful vs 79 helpful changes (p=0.20), so we should keep that conclusion as inconclusive. TGIF-QA is visual QA without audio or speaker labels, so it cannot be the main evidence for our speaker-attribution questions.

Please reuse the runner for a small AVA-ActiveSpeaker and AMI validation batch, followed by EasyCom transfer. For AVA, compare speaking-state predictions using unchanged synchronized audio. For AMI, verify participant mapping and ask which visible person spoke a real annotated utterance. Use captured, restored, and matched-resolution control inputs; record quality scores, introduced errors, corrections, agent answers, and abstentions. Keep tool-only and direct audiovisual models separate and record exact versions. Freeze the clips before inference and return all outcomes, including unchanged and improved cases. Please also make cache keys configuration-aware and audit ambiguous QA scoring before extending the TGIF experiment. We should scale the audiovisual pipeline once media/labels/scoring are validated, without requiring a positive failure result to select the data.

## Sources

- [Reviewed commit and experiment report](https://github.com/bolledduu/agent-audio-visual/blob/376950acaccb42d02736bf65ae0c4095d524c667/TGIFQA_RESULTS.md)
- [Reviewed implementation](https://github.com/bolledduu/agent-audio-visual/blob/376950acaccb42d02736bf65ae0c4095d524c667/scripts/tgifqa_restoration.py)
- [TGIF-QA official dataset](https://github.com/YunseokJANG/tgif-qa)
- [AVA official labels and downloads](https://sites.research.google/gr/ava/download/)
- [AMI audiovisual signal documentation](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml)
- [EasyCom dataset, download instructions, and known issues](https://github.com/facebookresearch/EasyComDataset)
- [VoxConverse annotations](https://github.com/joonson/voxconverse)

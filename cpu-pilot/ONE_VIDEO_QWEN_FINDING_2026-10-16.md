# One-video Qwen test: where the agent actually fails

**Qwen makes a repeatable speaker-attribution error on this clip, but restoration does not introduce the error.** It is already wrong with captured evidence. This is a negative result for the primary restoration-induced majority-answer hypothesis on this selected clip.

## The video and why it was chosen

Source `EEey6Pt0rqI`, face track `EEey6Pt0rqI:2`, source interval 7.28–8.28 seconds. The visible person moves their mouth. All 25 source CSV annotations in the interval say `NOT_SPEAKING`; these dataset labels are the evaluation target, not TalkNet predictions. They were checked against the archived annotation hash. No new independent human annotation was performed, and the images alone cannot establish who produced the audio.

This clip was selected deliberately because the preceding controlled experiment found a TalkNet error after restoration. It is a targeted diagnostic, not a random sample or independent validation set. Noise sigma 40, the original soundtrack, the one-second interval, and the restoration checkpoint were kept unchanged. This is synthetic Gaussian noise, not a measured Wi-Fi outage.

FastDVDnet reduced the noise. TalkNet's audiovisual classifier predicted not speaking on every captured frame, but speaking on the first five restored frames. The source-reference predictions were also not speaking throughout. Quality scores improved: MUSIQ 13.74 to 15.96; BRISQUE 62.78 to 61.10, where lower is better. This motivated asking whether Qwen would turn a tool error into an agent error.

## Actual Qwen experiment

Executed nine fresh agent sessions using `qwen/qwen3.8-27b` through Groq: three repetitions of each condition. The run made 21 completion requests. The prompts, tool responses, structured answers, model identifiers, API response identifiers, usage records, and evaluation are saved.

The primary question was:

> For this one-second clip, is the visible person speaking for the majority of the 24 scored timestamps?

The same request also asked for a tag at each timestamp: speaking, not speaking, or uncertain. This was declared a secondary outcome before any model calls. Twenty-four timestamps are scored because of the existing audio/video feature alignment; the final video frame is not scored.

Conditions:

1. **Captured evidence:** tools expose captured-video measurements only.
2. **Restored evidence:** the same tools expose actual FastDVDnet-output measurements only.
3. **Restored plus captured comparison:** a separate condition adds a tool to inspect captured measurements and their disagreement with current evidence.

The primary captured/restored pair has identical model, temperature, system/user prompts, tool schemas, and original audio. Configuration hashes match. Current-evidence measurements are the only changed input. Repeat order is counterbalanced. Temperature is zero; repetitions assess run stability, not performance on independent videos.

Qwen receives **tool measurements, not pixels or audio samples**. Tools provide audiovisual, visual-only, and audio-only scores plus visual-quality scores. They never provide source IDs, annotations, accuracy, source-reference images, or reference metrics. A label-informed evaluator scores answers only after submission.

Unlike the earlier eight-case selection pilot, this diagnostic also exposes the audio-only score series. It is a new, documented agent environment rather than a replication of the earlier 7/8 majority-answer result. The extra field is present in both sides of the primary pair.

## Results

The dataset says the visible person is not speaking throughout.

| Evidence available to Qwen | Majority answer, all three runs | Correct majority answers | Wrong timestamp tags per run | Uncertain timestamps per run |
|---|---|---:|---|---|
| Captured only | Speaking | 0/3 | 24, 24, 24 | 0, 0, 0 |
| Restored only | Speaking | 0/3 | 24, 24, 24 | 0, 0, 0 |
| Restored plus captured comparison | Not speaking | 3/3 | 5, 5, 5 | 19, 0, 19 |

**Primary paired outcome: zero correct-to-wrong majority-answer transitions out of three matched pairs.** The timestamped error count is also unchanged in those pairs. The captured-only answer is already wrong, so the restored-only error cannot be attributed to restoration.

Self-reported confidence is 0.75 with captured evidence and 0.70 with restored evidence in all repetitions. It does not increase after restoration and is not calibrated confidence. Comparison-condition confidence is 0.45, 0.55, and 0.45.

The comparison condition improves the majority answer on this one case. It does not eliminate timestamped errors. Its five false speaking tags occur at relative timestamps 0.00, 0.04, 0.08, 0.12, and 0.16 seconds—the same initial locations where restored TalkNet scores cross 0.5. Runs 1 and 3 cover only 5/24 timestamps, and all five answered timestamps are wrong; run 2 covers all 24, with five wrong and nineteen correct. Do not call the comparison condition a uniformly safe or validated safeguard, and do not compare its pooled frame error without reporting coverage.

## Where Qwen's reasoning goes wrong

One captured-evidence answer explains:

> The combined audio-visual scores are lower but the strong, consistent audio signal supports that the visible person is speaking throughout the clip.

That conclusion is unsupported. The audio-only head receives no visual identity information. Its score cannot establish that the person in this particular face crop is producing the speech. Those audio-only scores are identical before and after restoration. Qwen nevertheless prioritizes them over the low audiovisual scores and assigns speaking to the visible person throughout.

In simple terms, the agent treats **“the audio suggests speech” as “this visible person is speaking.”** The trace is consistent with that evidence-fusion mistake. It is not a causal ablation proving the model's internal mechanism; no audio-field removal or tool-description intervention was run.

When a captured-comparison tool is available, Qwen explicitly notes that the high audio-only scores do not identify the visible speaker, and answers the majority question correctly. However, it continues to accept the five initial restored speaking tags. Two repetitions abstain on the other nineteen timestamps. The tool comparison helps one answer while leaving a finer-grained failure.

The original tool description identified the scores as audio-only but did not explicitly explain their inability to identify a face. Tool documentation is therefore a plausible contributing factor. A generic clarification of that limitation would be an engineering baseline, not evidence of a novel research method. The current traces do not justify assigning the entire failure to restoration or to the model independently of its tool interface.

## What this says about our research questions

- **RQ1, restoration-induced Qwen error:** not demonstrated here. The agent is wrong before and after restoration. The clip does demonstrate a real Qwen error, separate from the TalkNet error.
- **RQ2, assessing misleading evidence:** comparison changes Qwen's explanation and majority answer, but it does not reliably resolve the conflicting timestamped evidence. No new assessor is validated.
- **RQ3, safe use of restoration:** the simple comparison tool has a measured benefit on the majority question in this case, but retains errors and sometimes greatly reduces timestamp coverage. It is not a complete safeguard or a novel method.

This test does not support the claim “restoration makes Qwen more confidently wrong.” It also does not universally disprove that claim. A one-clip diagnosis should not be promoted into a full-paper recommendation.

## Inspectable evidence

- [Video, timeline, exact answers, and tool traces](data/one_video_qwen_review/index.html)
- [Frame-by-frame result figure](results/one_video_qwen_timeline.png)
- [Actual API-run results](results/one_video_qwen_paired.json)
- [Protocol and exact prompts](one_video_qwen_protocol.json)
- [Independent arithmetic and payload audit](results/one_video_qwen_audit.json)
- [Runner](scripts/one_video_qwen_paired.py)

The video preview displays **TalkNet** overlays. The separate timeline and recorded submissions show **Qwen** outputs. Keeping those two clearly labelled is essential to avoid repeating the earlier attribution mistake.

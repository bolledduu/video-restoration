# Fixed three-video Qwen pilot â€” 16 October 2026

**Decision: DO NOT commit to the full restoration-induced Qwen majority-error paper from this pilot.**

This is the fixed decision for the specified tool-agent and test, not a universal rejection of audiovisual restoration research. Three sources, five one-second person-tracks, two repeats per captured/restored pair: 20 actual Qwen sessions. Repeated calls are not independent videos.

## Question and setup

Does replacing degraded-video measurements with measurements from actual FastDVDnet restorations turn a correct Qwen speaker answer into a wrong one? Qwen receives TalkNet audiovisual scores, audio-only activity scores, and MUSIQ/BRISQUE scores through tools. It does not receive video pixels or audio waveforms. This measures an agent using perception tools, not a video-native Qwen model.

Source video, interval, audio, question, tools, temperature, and model are held fixed within each pair. The tool explicitly explains that audio activity cannot identify the visible speaker. This interface correction and removal of the visual-only head were fixed before this run; the previous ambiguous-interface experiment is not pooled with it.

FastDVDnet outputs and measurements were reused from prior real CPU runs, with output/source/annotation hashes checked. Qwen calls are fresh. No audio-conditioned restorer was tested. Labels come from UniTalk-ASD, not fresh independent human adjudication.

## Frozen decision rule

Proceed to larger restoration-induced Qwen majority-error study only if repeated primary harm occurs on at least two distinct source videos.

This is an operational pilot screen, not a statistical significance test. Cases were intentionally selected from existing tracks, including known detector failures. Neither prevalence nor generalization can be estimated from this sample.

## Results (each number is repeat 1 / repeat 2)

| Source and track | Dataset label | Captured majority correct | Restored majority correct | Wrong tags captured â†’ restored | Abstentions captured â†’ restored |
|---|---|---|---|---|---|
| kgVz_AXYvn0, 1.00s, kgVz_AXYvn0:3 | not speaking | 1/1 | 1/1 | 0/0 â†’ 0/0 | 24/24 â†’ 24/24 |
| kgVz_AXYvn0, 9.32s, kgVz_AXYvn0:18 | speaking | 1/1 | 0/0 | 0/0 â†’ 0/0 | 0/0 â†’ 24/24 |
| EEey6Pt0rqI, 7.28s, EEey6Pt0rqI:2 | not speaking | 1/1 | 1/1 | 0/0 â†’ 5/5 | 0/0 â†’ 0/0 |
| EEey6Pt0rqI, 10.72s, EEey6Pt0rqI:9 | speaking | 1/1 | 1/1 | 0/0 â†’ 0/0 | 0/0 â†’ 0/0 |
| 9Zja_cIAwY8, 0.00s, 9Zja_cIAwY8:0 | not speaking | 1/1 | 1/1 | 0/0 â†’ 0/0 | 0/0 â†’ 24/24 |

Primary transitions: **2/10 correct-to-wrong pairs**, **0/10 wrong-to-correct pairs**. Repeatable harmful source count: **1/3**.

All timestamp counts use 24 scored timestamps per session. Zero wrong tags with 24 abstentions means no timestamp predictions were supplied; it is not perfect timestamp accuracy. The legacy raw field `false_speaking_indices` lists all predicted-speaking indices, including correct ones on speaking tracks; it is not used to count errors. Audited errors compare tags directly with labels.

## Detector baseline and quality

| Track | TalkNet captured errors /24 | TalkNet restored errors /24 | Reference-video errors /24 | MUSIQ captured â†’ restored | BRISQUE captured â†’ restored |
|---|---:|---:|---:|---:|---:|
| kgVz_AXYvn0__label0__sigma40 | 3 | 0 | 3 | 14.74 â†’ 16.02 | 62.57 â†’ 57.25 |
| kgVz_AXYvn0__label1__sigma40 | 0 | 24 | 24 | 16.32 â†’ 15.40 | 60.57 â†’ 51.26 |
| EEey6Pt0rqI__label0__sigma40 | 0 | 5 | 0 | 13.74 â†’ 15.96 | 62.78 â†’ 61.10 |
| EEey6Pt0rqI__label1__sigma40 | 0 | 0 | 0 | 15.72 â†’ 16.20 | 62.95 â†’ 51.38 |
| 9Zja_cIAwY8 | 0 | 0 | 0 | 15.47 â†’ 18.67 | 50.29 â†’ 51.01 |

A restored error on a track where the reference detector also fails does not establish hallucinated evidence. It may remove noise that accidentally helped the detector. Improved visual-quality scores do not establish a mechanism.

## How to interpret the two outcomes

The majority question asks whether the person speaks for most of a second. The timestamp question asks whether the person speaks at each of 24 instants. Five incorrect speaking tags can therefore coexist with a correct majority answer. Both are real outcomes, but the timestamp outcome cannot be used to retroactively pass the majority-answer expansion rule.

These traces can demonstrate a failure of the tool-agent pipeline. They cannot establish that Qwen visually perceived a fabricated detail, that restoration was audio-conditioned, that an agent made a wrong quotation or real-world action, or that a proposed safeguard is effective. No safeguard was evaluated in this fixed comparison.

An earlier run at a 1600-token output cap was interrupted by the provider limit after four sessions. It is retained separately and excluded. All final sessions use the uniform 800-token cap; some submitted explanation strings end abruptly, but the submitted structured answer fields were valid.

## Every Qwen answer and explanation

### kgVz_AXYvn0__label0__sigma40 â€” captured â€” repeat 1

Majority speaking: False; confidence: 0.4; wrong tags: 0; abstentions: 24.

The audio-visual speaking scores for this visible person stay below the 0.5 detector threshold for all 24 timestamps (range ~0.32â€“0.54, with only a few late frames marginally above 0.5). The audio-only activity is consistently high (~0.58â€“0.75), indicating speech is present, but it cannot be attributed to the visible person. Since the AV scores are mostly below threshold and only a few late frames are borderline, the visible person is not speaking for the majority of the clip. Given the uncl

### kgVz_AXYvn0__label0__sigma40 â€” restored â€” repeat 1

Majority speaking: False; confidence: 0.5; wrong tags: 0; abstentions: 24.

The audio-visual speaking scores for the visible person range from ~0.27 to ~0.48, all below the 0.5 detector threshold, so the detector does not indicate this person is speaking at any timestamp. The audio-only activity scores are higher (~0.58-0.75), suggesting some audio activity, but that head cannot identify the visible speaker and may reflect another or offscreen person. Because the AV scores are consistently below threshold but not extremely low, I mark all timestamps as uncertain (-1

### kgVz_AXYvn0__label0__sigma40 â€” restored â€” repeat 2

Majority speaking: False; confidence: 0.5; wrong tags: 0; abstentions: 24.

The audio-visual speaking scores for the visible person range from ~0.27 to ~0.48, all below the 0.5 detector threshold, so the detector does not indicate this person is speaking at any timestamp. The audio-only activity scores are higher (~0.58-0.75), suggesting some audio activity, but that head cannot identify the visible speaker and may reflect another or offscreen person. Because the AV scores are consistently below threshold but not extremely low, I mark all timestamps as uncertain (-1

### kgVz_AXYvn0__label0__sigma40 â€” captured â€” repeat 2

Majority speaking: False; confidence: 0.4; wrong tags: 0; abstentions: 24.

The audio-visual speaking scores for this visible person stay below the 0.5 detector threshold for all 24 timestamps (range ~0.32â€“0.54, with only a few late frames marginally above 0.5). The audio-only activity is consistently high (~0.58â€“0.75), indicating speech is present, but it cannot be attributed to the visible person. Since the AV scores are mostly below threshold and only a few late frames are borderline, the visible person is not speaking for the majority of the clip. Given the uncl

### kgVz_AXYvn0__label1__sigma40 â€” captured â€” repeat 1

Majority speaking: True; confidence: 0.7; wrong tags: 0; abstentions: 0.

The audio-visual speaking scores for the visible person are above the 0.5 detector threshold at all 24 timestamps (range ~0.51-0.67), indicating continuous speaking. Audio-only activity is also present throughout, consistent with speech. I tag all timestamps as speaking (1) since the AV detector consistently exceeds threshold. Confidence is moderate because scores are uncalibrated and the audio-only head cannot confirm the visible person is the speaker.

### kgVz_AXYvn0__label1__sigma40 â€” restored â€” repeat 1

Majority speaking: False; confidence: 0.55; wrong tags: 0; abstentions: 24.

The audio-visual speaking scores for the visible person are all well below the 0.5 detector threshold (range ~0.20â€“0.33), indicating the visible person is not clearly speaking. The audio-only activity is elevated (~0.56â€“0.78), but that head cannot identify the speaker and may reflect another/offscreen person. Since the AV scores never cross threshold, the visible person is not speaking for the majority of timestamps; I mark all as uncertain (-1) given the uncalibrated, sub-threshold signals.

### kgVz_AXYvn0__label1__sigma40 â€” restored â€” repeat 2

Majority speaking: False; confidence: 0.55; wrong tags: 0; abstentions: 24.

The audio-visual speaking scores for the visible person are all well below the 0.5 detector threshold (range ~0.20â€“0.33), indicating the visible person is not clearly speaking. The audio-only activity is elevated (~0.56â€“0.78), but that head cannot identify the speaker and may reflect another/offscreen person. Since the AV scores never cross threshold, the visible person is not speaking for the majority of timestamps; I mark all as uncertain (-1) given the uncalibrated, sub-threshold signals.

### kgVz_AXYvn0__label1__sigma40 â€” captured â€” repeat 2

Majority speaking: True; confidence: 0.6; wrong tags: 0; abstentions: 0.

The audio-visual speaking scores for the visible person are consistently above the 0.5 detector threshold across all 24 timestamps (range ~0.51â€“0.67), and the audio-only activity is also elevated throughout (~0.56â€“0.78). Since the AV head measures this person specifically and stays above threshold for the entire clip, the visible person is speaking for the majority of timestamps. Confidence is moderate because scores are uncalibrated and the audio-only head cannot confirm the visible person,

### EEey6Pt0rqI__label0__sigma40 â€” captured â€” repeat 1

Majority speaking: False; confidence: 0.7; wrong tags: 0; abstentions: 0.

The audio-visual speaking scores for the visible person are consistently low (0.22â€“0.35), all well below the 0.5 detector threshold, indicating the visible person is not speaking. The audio-only activity scores are higher (0.56â€“0.80), suggesting audio activity is present, but this head cannot identify the speaker and may reflect another or offscreen person. Since the person-specific scores never cross threshold, the visible person is not speaking for the majority of timestamps.

### EEey6Pt0rqI__label0__sigma40 â€” restored â€” repeat 1

Majority speaking: False; confidence: 0.55; wrong tags: 5; abstentions: 0.

The audio-visual speaking scores for this visible person start just above the 0.5 threshold (0.56-0.60) for the first ~5 timestamps, then steadily decline below 0.5 (0.49 down to ~0.29-0.33) for the rest of the clip. The audio-only activity remains moderately high throughout (0.56-0.80), but it cannot identify the visible speaker and may reflect another/offscreen source. Since the AV scores are below threshold for 19 of 24 timestamps, the visible person is not speaking for the majority.

### EEey6Pt0rqI__label0__sigma40 â€” restored â€” repeat 2

Majority speaking: False; confidence: 0.55; wrong tags: 5; abstentions: 0.

The audio-visual speaking scores for this visible person start just above the 0.5 threshold (0.56-0.60) for the first ~5 timestamps, then steadily decline below 0.5 (0.49 down to ~0.29-0.33) for the rest of the clip. The audio-only activity remains moderately high throughout (0.56-0.80), but it cannot identify the visible speaker and may reflect another/offscreen source. Since the AV scores are below threshold for 19 of 24 timestamps, the visible person is not speaking for the majority.

### EEey6Pt0rqI__label0__sigma40 â€” captured â€” repeat 2

Majority speaking: False; confidence: 0.7; wrong tags: 0; abstentions: 0.

The audio-visual speaking scores for the visible person are consistently low (roughly 0.22â€“0.35), well below the 0.5 detector threshold across all 24 timestamps, indicating the visible person is not speaking. The audio-only activity is higher (0.56â€“0.80), but that head cannot identify the speaker and may reflect another/offscreen person. Since the person-specific scores never cross threshold, the majority answer is not speaking, with all tags set to 0.

### EEey6Pt0rqI__label1__sigma40 â€” captured â€” repeat 1

Majority speaking: True; confidence: 0.85; wrong tags: 0; abstentions: 0.

The audio-visual speaking scores for the visible person are consistently very high (0.95â€“0.999) across all 24 timestamps, well above the 0.5 detector threshold, indicating the visible person is speaking throughout the clip. The audio-only activity scores are moderate (~0.6â€“0.77) and consistent with ongoing speech, though they cannot confirm speaker identity. No timestamps appear uncertain, so all are tagged 1.

### EEey6Pt0rqI__label1__sigma40 â€” restored â€” repeat 1

Majority speaking: True; confidence: 0.85; wrong tags: 0; abstentions: 0.

The audio-visual speaking scores for the visible person are consistently high (0.85â€“0.999) across all 24 timestamps, well above the 0.5 detector threshold, indicating the visible person is speaking throughout the clip. Audio-only activity is also present (~0.6â€“0.77), consistent with ongoing speech. Image quality is moderate but does not affect speaker attribution.

### EEey6Pt0rqI__label1__sigma40 â€” restored â€” repeat 2

Majority speaking: True; confidence: 0.85; wrong tags: 0; abstentions: 0.

The audio-visual speaking scores for the visible person are consistently high (0.85â€“0.999) across all 24 timestamps, well above the 0.5 detector threshold, indicating the visible person is speaking throughout the clip. Audio-only activity is also present (~0.6â€“0.77), consistent with ongoing speech. Image quality is moderate but does not affect speaker attribution.

### EEey6Pt0rqI__label1__sigma40 â€” captured â€” repeat 2

Majority speaking: True; confidence: 0.85; wrong tags: 0; abstentions: 0.

The audio-visual speaking scores for the visible person are consistently very high (0.95â€“0.999) across all 24 timestamps, well above the 0.5 detector threshold, indicating the visible person is speaking throughout the clip. The audio-only activity scores are moderate (~0.6â€“0.77) and consistent with ongoing speech, though they cannot confirm speaker identity. Image quality metrics are neutral and do not affect speaker attribution. Based on the strong, consistent audio-visual evidence, the tag

### 9Zja_cIAwY8 â€” captured â€” repeat 1

Majority speaking: False; confidence: 0.7; wrong tags: 0; abstentions: 0.

The audio-visual speaking scores for this visible person are all very low (0.026â€“0.101), far below the 0.5 detector threshold, indicating the visible person is not speaking. The audio-only activity is elevated (0.39â€“0.85), suggesting someone is speaking, but that head cannot identify the speaker and may be another/offscreen person. Since the AV scores for this person never approach the threshold, the visible person is not speaking for the majority of timestamps.

### 9Zja_cIAwY8 â€” restored â€” repeat 1

Majority speaking: False; confidence: 0.5; wrong tags: 0; abstentions: 24.

The audio-visual speaking scores for this visible person stay low (0.11â€“0.24), well below the 0.5 detector threshold across all 24 timestamps, indicating the visible person is not clearly speaking. The audio-only activity is higher (0.39â€“0.85), but that head cannot identify the speaker and may reflect another/offscreen person. Since the AV scores never cross threshold and are uncalibrated, I mark all timestamps uncertain (-1) and conclude the visible person is not speaking for the majority.

### 9Zja_cIAwY8 â€” restored â€” repeat 2

Majority speaking: False; confidence: 0.5; wrong tags: 0; abstentions: 24.

The audio-visual speaking scores for this visible person stay low (0.11â€“0.24), well below the 0.5 detector threshold across all 24 timestamps, indicating the visible person is not clearly speaking. The audio-only activity is higher (0.39â€“0.85), but that head cannot identify the speaker and may reflect another/offscreen person. Since the AV scores never cross threshold and are uncalibrated, I mark all timestamps uncertain (-1) and conclude the visible person is not speaking for the majority.

### 9Zja_cIAwY8 â€” captured â€” repeat 2

Majority speaking: False; confidence: 0.7; wrong tags: 0; abstentions: 0.

The audio-visual speaking scores for this visible person are very low across all 24 timestamps (max ~0.10, well below the 0.5 detector threshold), indicating the visible person is not speaking. The audio-only activity is elevated (0.39â€“0.85), but that head cannot identify the speaker and may reflect another/offscreen person. Since the person-specific scores are uniformly low, I tag all timestamps as not speaking (0) and conclude the visible person is not speaking for the majority of the clip

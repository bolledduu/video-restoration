# Four-RQ source-transfer pilot — 16 October 2026

**Focused method-development screen: FAIL.** This is not validation of all four RQs, novelty, or paper readiness.

Two additional source videos; the earliest eligible speaking and non-speaking track from each, selected before inference. Four tracks, real CPU FastDVDnet and spatial NLM outputs, fixed Gaussian noise sigma 25. Twenty-four actual Qwen tool-agent sessions: three evidence conditions × four tracks × two repeats. No severity or sample search after outcomes. Labels/reference images stay outside the agent tools.

The source videos are new to the paired Qwen experiment, but some tracks were already used in detector-only experiments. This is source transfer from the prior Qwen pilot, not an untouched external validation set. Qwen consumes detector/quality measurements, not pixels or waveforms.

## RQ1 — failure replication

| Track | Restorer | Captured wrong tags (r1/r2) | Restored wrong tags (r1/r2) | New errors on common answered timestamps | Corrections on common timestamps | Overall correct to wrong (r1/r2) | Both quality scores improve? |
|---|---|---:|---:|---:|---:|---|---|
| ABqo634gI5s, label 0, 55.32s | fastdvdnet | 0/0 | 0/0 | 0/0 | 0/0 | False/False | False |
| ABqo634gI5s, label 0, 55.32s | spatial_nlm | 0/0 | 0/0 | 0/0 | 0/0 | False/False | False |
| ABqo634gI5s, label 1, 71.24s | fastdvdnet | 0/0 | 0/0 | 0/0 | 0/0 | False/False | True |
| ABqo634gI5s, label 1, 71.24s | spatial_nlm | 0/0 | 0/0 | 0/0 | 0/0 | False/False | False |
| qv3-HaaxGUc, label 0, 14.52s | fastdvdnet | 0/0 | 0/0 | 0/0 | 0/0 | False/False | True |
| qv3-HaaxGUc, label 0, 14.52s | spatial_nlm | 0/0 | 0/0 | 0/0 | 0/0 | False/False | True |
| qv3-HaaxGUc, label 1, 14.84s | fastdvdnet | 6/6 | 24/24 | 18/18 | 0/0 | True/True | True |
| qv3-HaaxGUc, label 1, 14.84s | spatial_nlm | 6/6 | 15/0 | 9/0 | 0/0 | True/True | True |

Repeatable primary-harm sources: ['qv3-HaaxGUc']. Repeatable corrected tracks: [].

Wrong-label counts exclude abstentions. A zero count does not establish correctness if no timestamps were answered; coverage is reported below. Repeats are not independent videos.

| Track | Evidence | MUSIQ (higher better) | BRISQUE (lower better) | TalkNet errors /24 |
|---|---|---:|---:|---:|
| ABqo634gI5s, label 0 | captured | 15.561 | 51.930 | 0 |
| ABqo634gI5s, label 0 | spatial_nlm | 13.783 | 46.514 | 0 |
| ABqo634gI5s, label 0 | fastdvdnet | 18.810 | 52.043 | 0 |
| ABqo634gI5s, label 0 | reference | 17.999 | 58.971 | 0 |
| ABqo634gI5s, label 1 | captured | 15.129 | 54.945 | 24 |
| ABqo634gI5s, label 1 | spatial_nlm | 14.813 | 44.746 | 24 |
| ABqo634gI5s, label 1 | fastdvdnet | 17.324 | 50.684 | 24 |
| ABqo634gI5s, label 1 | reference | 16.014 | 58.262 | 23 |
| qv3-HaaxGUc, label 0 | captured | 15.403 | 49.900 | 0 |
| qv3-HaaxGUc, label 0 | spatial_nlm | 21.652 | 6.598 | 0 |
| qv3-HaaxGUc, label 0 | fastdvdnet | 28.257 | 48.999 | 0 |
| qv3-HaaxGUc, label 0 | reference | 27.765 | 43.543 | 0 |
| qv3-HaaxGUc, label 1 | captured | 14.386 | 46.288 | 6 |
| qv3-HaaxGUc, label 1 | spatial_nlm | 16.415 | 1.410 | 18 |
| qv3-HaaxGUc, label 1 | fastdvdnet | 20.194 | 46.245 | 24 |
| qv3-HaaxGUc, label 1 | reference | 18.855 | 43.772 | 24 |

On the strongest repeated new regression (`qv3-HaaxGUc:6`, 14.84–15.84s), FastDVDnet changes Qwen errors from 6 to 24 with full coverage in both repeats. MUSIQ rises from 14.386 to 20.194; BRISQUE decreases only slightly, from 46.288 to 46.245. The latter is a numerical improvement, not evidence of a meaningful or statistically significant quality gain. Reference-video TalkNet also has 24/24 errors, so the result does not establish invented mouth details; restoration can remove noise that happened to help an already unreliable detector.

Spatial NLM on that same track gives 15 errors/21 answered timestamps in repeat 1, but 0 errors/8 answered in repeat 2. Its overall answer is wrong in both. Do not present the smaller error count as improved accuracy: coverage and answers changed substantially even at temperature zero.

## RQ2 — assessment baselines

There are 8 track/restorer candidates, of which 1 have repeated new errors and 0 have repeated corrections. Only 5 candidates have at least one jointly answered timestamp in both repeats and are eligible for the observed-error ranking analysis. No common answers is not evidence of preserved attribution. Risk scores are averaged over repeats; repeats are not treated as independent candidates.

| Risk score | Harm vs other AUROC | Harm vs correction AUROC |
|---|---:|---:|
| low_musiq | 0.500 | Not estimable (missing class) |
| high_brisque | 0.500 | Not estimable (missing class) |
| low_qwen_confidence | 0.375 | Not estimable (missing class) |
| detector_disagreement | 1.000 | Not estimable (missing class) |
| low_detector_confidence | 0.500 | Not estimable (missing class) |

Harm-vs-correction excludes candidates with neither outcome or both outcomes. AUC, where calculable, is descriptive on this tiny known-design sample; no confidence or superiority claim is supported. No new assessor was trained. These baselines cannot establish the RQ2 claim of performance beyond existing baselines.

## RQ3 — frozen guard transfer

| Restorer | Policy | Wrong timestamp tags | Answered /192 | Wrong overall / answered |
|---|---|---:|---:|---:|
| fastdvdnet | captured_only | 12 | 144/192 | 2/8 |
| fastdvdnet | restored_only | 48 | 144/192 | 4/8 |
| fastdvdnet | detector_disagreement_abstain | 12 | 108/192 | 2/6 |
| fastdvdnet | detector_disagreement_capture | 12 | 144/192 | 2/8 |
| fastdvdnet | captured_answer_priority | 12 | 144/192 | 2/8 |
| spatial_nlm | captured_only | 12 | 144/192 | 2/8 |
| spatial_nlm | restored_only | 15 | 77/192 | 4/8 |
| spatial_nlm | detector_disagreement_abstain | 6 | 66/192 | 2/6 |
| spatial_nlm | detector_disagreement_capture | 6 | 90/192 | 2/8 |
| spatial_nlm | captured_answer_priority | 12 | 144/192 | 2/8 |

Captured-only rows are the same eight Qwen sessions reused as the comparator for both restoration families. Do not sum them as independent observations. Guards use the exact previous fixed rules and no labels. They route answers, not image pixels. This does not measure perceptual quality of a novel image-selection framework. Common-answered comparisons are saved in the audit JSON; they restrict to timestamps answered by both systems and are not a full matched-coverage risk curve.

## RQ4 — what transferred

The frozen guards and unchanged Qwen prompts were applied to both additional sources and two restoration families. This tests a limited transfer slice. Only one agent model, one dataset and one noise severity were run. Generalization across unseen agent models, datasets and degradation types remains untested; no framework exists to validate on all those axes.

## Investment decision

Before inference, the screening rule required both repeatable quality-improving harm on a new source and a repeatable correction of a captured Qwen error on some track. This rule tests whether there is evidence for a useful restoration-versus-risk tradeoff. The earlier majority-answer screen remains unchanged; this timestamp screen is a separately declared follow-up.

The screen fails: these additional cases do not justify launching the full framework experiment campaign under this fixed screen. This does not erase the earlier positive failure example and does not universally disprove the questions.

Both restorers are video-only. None of these results establishes audio-created visual details.

## Exact agent explanations

### ABqo634gI5s__label0__sigma25 / captured / repeat 1

Majority speaking: False; confidence: 0.7; evaluation: {'majority_correct': True, 'frame_wrong': 0, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

The audio-visual speaking scores for the visible person are consistently low (0.04–0.14), well below the 0.5 detector threshold across all 24 timestamps, indicating the visible person is not speaking. The audio-only activity scores are higher (0.5–0.76), suggesting some audio activity, but that head cannot identify the speaker and may reflect another or offscreen person. Based on the audio-visual head, the visible person is not speaking for the majority of the clip.

### ABqo634gI5s__label0__sigma25 / fastdvdnet / repeat 1

Majority speaking: False; confidence: 0.7; evaluation: {'majority_correct': True, 'frame_wrong': 0, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

The audio-visual speaking scores for the visible person are all well below the 0.5 detector threshold (ranging ~0.15-0.32), indicating the visible person is not speaking. The audio-only activity scores are higher (~0.5-0.76), suggesting some audio activity, but this head cannot identify the speaker and may reflect another/offscreen person. Since the person-specific scores are consistently low, the visible person is not speaking for the majority of timestamps.

### ABqo634gI5s__label0__sigma25 / spatial_nlm / repeat 1

Majority speaking: False; confidence: 0.55; evaluation: {'majority_correct': True, 'frame_wrong': 0, 'frame_abstentions': 24, 'frame_coverage': 0.0, 'false_speaking_indices': []}

The audio-visual speaking scores for the visible person are all below the 0.5 detector threshold (ranging ~0.17–0.40), trending downward, suggesting the visible person is not speaking for the majority of the clip. However, the audio-only activity scores are consistently high (~0.51–0.76), indicating someone is speaking in the audio — possibly an offscreen or different person. Since the audio-visual head (which concerns this person) is below threshold throughout, I judge the visible person is

### ABqo634gI5s__label0__sigma25 / spatial_nlm / repeat 2

Majority speaking: False; confidence: 0.55; evaluation: {'majority_correct': True, 'frame_wrong': 0, 'frame_abstentions': 24, 'frame_coverage': 0.0, 'false_speaking_indices': []}

The audio-visual speaking scores for this visible person are all below the 0.5 detector threshold (ranging ~0.17–0.40), suggesting the visible person is not speaking for the majority of the clip. However, the audio-only activity scores are consistently high (~0.51–0.76), indicating someone is speaking in the audio — possibly an offscreen or different person. Since the audio-visual scores are below threshold but not extremely low, and the audio-only head cannot attribute speech to the visible

### ABqo634gI5s__label0__sigma25 / fastdvdnet / repeat 2

Majority speaking: False; confidence: 0.7; evaluation: {'majority_correct': True, 'frame_wrong': 0, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

The audio-visual speaking scores for the visible person are consistently low (0.15–0.32), all well below the 0.5 detector threshold, indicating the visible person is not speaking. The audio-only activity scores are higher (0.5–0.76), suggesting some audio activity, but this head cannot identify the speaker and may reflect another or offscreen person. Since the AV scores for this person never approach the threshold, the majority answer is not speaking.

### ABqo634gI5s__label0__sigma25 / captured / repeat 2

Majority speaking: False; confidence: 0.7; evaluation: {'majority_correct': True, 'frame_wrong': 0, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

The audio-visual speaking scores for the visible person are consistently low (0.04–0.14), well below the 0.5 detector threshold across all 24 timestamps, indicating the visible person is not speaking. The audio-only activity scores are moderate (0.5–0.76), suggesting some audio activity, but this head cannot attribute speech to the visible person and may reflect another/offscreen speaker. Based on the low AV scores, the visible person is not speaking for the majority of the clip.

### ABqo634gI5s__label1__sigma25 / captured / repeat 1

Majority speaking: False; confidence: 0.55; evaluation: {'majority_correct': False, 'frame_wrong': 0, 'frame_abstentions': 24, 'frame_coverage': 0.0, 'false_speaking_indices': []}

The audio-visual speaking scores for this visible person are all below the 0.5 detector threshold (ranging ~0.15–0.46), trending downward, so the detector does not indicate this person is speaking for the majority of the 24 timestamps. The audio-only activity is moderately high (~0.6–0.78), but that head cannot identify the visible speaker and may reflect another/offscreen person. Given the consistently sub-threshold AV scores, I judge the visible person is not speaking for the majority. I'm

### ABqo634gI5s__label1__sigma25 / fastdvdnet / repeat 1

Majority speaking: False; confidence: 0.5; evaluation: {'majority_correct': False, 'frame_wrong': 0, 'frame_abstentions': 24, 'frame_coverage': 0.0, 'false_speaking_indices': []}

The audio-visual speaking scores for the visible person range from ~0.29 to ~0.45, all below the 0.5 detector threshold, so the detector does not indicate this person is speaking at any timestamp. The audio-only activity scores are higher (~0.59-0.78), indicating audio activity, but that head cannot identify the visible speaker and may reflect another or offscreen person. Because the AV scores are consistently below threshold but not extremely low, I mark each timestamp as uncertain (-1) and

### ABqo634gI5s__label1__sigma25 / spatial_nlm / repeat 1

Majority speaking: False; confidence: 0.55; evaluation: {'majority_correct': False, 'frame_wrong': 0, 'frame_abstentions': 24, 'frame_coverage': 0.0, 'false_speaking_indices': []}

The audio-visual speaking scores for this visible person stay below the 0.5 detector threshold across all 24 timestamps (range ~0.30–0.47), suggesting the visible person is not clearly speaking. The audio-only activity is elevated (~0.59–0.78), but that head cannot identify the speaker and may reflect another/offscreen person. Given the AV scores never cross threshold and are uncalibrated, I mark all timestamps uncertain (-1) and conclude the visible person is not speaking for the majority.

### ABqo634gI5s__label1__sigma25 / spatial_nlm / repeat 2

Majority speaking: False; confidence: 0.55; evaluation: {'majority_correct': False, 'frame_wrong': 0, 'frame_abstentions': 24, 'frame_coverage': 0.0, 'false_speaking_indices': []}

The audio-visual speaking scores for the visible person stay below the 0.5 detector threshold across all 24 timestamps (range ~0.30–0.47), suggesting the visible person is not clearly speaking. The audio-only activity is elevated (~0.59–0.78), indicating some audio activity, but that head cannot attribute speech to the visible person and may reflect another/offscreen speaker. Given the AV scores never cross threshold and the audio-only signal is ambiguous for attribution, I mark all per-tim

### ABqo634gI5s__label1__sigma25 / fastdvdnet / repeat 2

Majority speaking: False; confidence: 0.5; evaluation: {'majority_correct': False, 'frame_wrong': 0, 'frame_abstentions': 24, 'frame_coverage': 0.0, 'false_speaking_indices': []}

The audio-visual speaking scores for this visible person range from ~0.29 to ~0.45, all below the 0.5 detector threshold, so the detector does not indicate this person is speaking at any timestamp. The audio-only activity scores are higher (~0.59-0.78), indicating audio activity, but that head cannot identify the visible speaker and may reflect another or offscreen person. Because the AV scores are consistently below threshold but not extremely low, I mark each timestamp as uncertain (-1) to

### ABqo634gI5s__label1__sigma25 / captured / repeat 2

Majority speaking: False; confidence: 0.55; evaluation: {'majority_correct': False, 'frame_wrong': 0, 'frame_abstentions': 24, 'frame_coverage': 0.0, 'false_speaking_indices': []}

The audio-visual speaking scores for this visible person are all below the 0.5 detector threshold (ranging ~0.15–0.46), trending downward, indicating the visible person is not clearly speaking. The audio-only activity is elevated (~0.6–0.78), but that head cannot identify the speaker and may reflect another/offscreen person. Since the AV scores never cross threshold and are uncalibrated, I mark all timestamps uncertain (-1) and conclude the visible person is not speaking for the majority of

### qv3-HaaxGUc__label0__sigma25 / captured / repeat 1

Majority speaking: False; confidence: 0.7; evaluation: {'majority_correct': True, 'frame_wrong': 0, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

The audio-visual speaking scores for this visible person are consistently low (0.04–0.13), well below the 0.5 detector threshold across all 24 timestamps, indicating the visible person is not speaking. The audio-only activity is high (~0.6–0.77), but that head cannot identify the speaker and may reflect another/offscreen person. Since the person-specific scores never approach the threshold, the majority answer is not speaking.

### qv3-HaaxGUc__label0__sigma25 / fastdvdnet / repeat 1

Majority speaking: False; confidence: 0.7; evaluation: {'majority_correct': True, 'frame_wrong': 0, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

The audio-visual speaking scores for this visible person are consistently very low (0.03–0.09), far below the 0.5 detector threshold across all 24 timestamps, indicating the visible person is not speaking. The audio-only activity is high (~0.6–0.77), but that head cannot identify the speaker and may reflect another/offscreen person. Since the person-specific scores are uniformly low, the majority answer is not speaking, with all tags set to 0.

### qv3-HaaxGUc__label0__sigma25 / spatial_nlm / repeat 1

Majority speaking: False; confidence: 0.7; evaluation: {'majority_correct': True, 'frame_wrong': 0, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

The audio-visual speaking scores for the visible person are consistently very low (0.04–0.08), far below the 0.5 detector threshold across all 24 timestamps, indicating the visible person is not speaking. The audio-only activity is high (~0.6–0.77), but that head cannot identify the speaker and may reflect another/offscreen person. Image quality scores are good but do not bear on speaker attribution. Based on the low AV scores, the visible person is not speaking for the majority of the clip.

### qv3-HaaxGUc__label0__sigma25 / spatial_nlm / repeat 2

Majority speaking: False; confidence: 0.7; evaluation: {'majority_correct': True, 'frame_wrong': 0, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

The audio-visual speaking scores for the visible person are consistently very low (0.04–0.08), far below the 0.5 detector threshold across all 24 timestamps, indicating the visible person is not speaking. The audio-only activity is high (~0.6–0.77), but that head cannot identify the speaker and may reflect another/offscreen person. Image quality scores are good but do not bear on speaker attribution. Based on the low AV scores, the visible person is not speaking for the majority of the clip.

### qv3-HaaxGUc__label0__sigma25 / fastdvdnet / repeat 2

Majority speaking: False; confidence: 0.7; evaluation: {'majority_correct': True, 'frame_wrong': 0, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

The audio-visual speaking scores for this visible person are consistently very low (0.03–0.09), far below the 0.5 detector threshold across all 24 timestamps, indicating the visible person is not speaking. The audio-only activity is high (~0.6–0.77), but that head cannot identify the speaker and may reflect another/offscreen person. Since the person-specific scores are uniformly low, the majority answer is not speaking, with all tags set to 0.

### qv3-HaaxGUc__label0__sigma25 / captured / repeat 2

Majority speaking: False; confidence: 0.7; evaluation: {'majority_correct': True, 'frame_wrong': 0, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

The audio-visual speaking scores for this visible person are consistently low (0.04–0.13), well below the 0.5 detector threshold across all 24 timestamps, indicating the visible person is not speaking. The audio-only activity is high (~0.6–0.77), but that head cannot identify the speaker and may reflect another/offscreen person. Since the person-specific scores never approach the threshold, the majority answer is not speaking.

### qv3-HaaxGUc__label1__sigma25 / captured / repeat 1

Majority speaking: True; confidence: 0.6; evaluation: {'majority_correct': True, 'frame_wrong': 6, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

Audio-visual scores exceed 0.5 at 14/24 timestamps, indicating the visible person is speaking for the majority. Audio-only activity is consistently high (0.64-0.79), confirming speech presence. Some timestamps near the threshold (0.41-0.55) introduce uncertainty, but the overall pattern supports majority speaking.

### qv3-HaaxGUc__label1__sigma25 / fastdvdnet / repeat 1

Majority speaking: False; confidence: 0.7; evaluation: {'majority_correct': False, 'frame_wrong': 24, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

The audio-visual speaking scores for the visible person are consistently low (range ~0.09–0.26), all well below the 0.5 detector threshold, indicating the visible person is not speaking. The audio-only activity scores are high (~0.64–0.79), suggesting someone is speaking, but this head cannot identify the speaker and may be an offscreen or different person. Since the AV scores for this person never approach the threshold, the visible person is not speaking for the majority of timestamps.

### qv3-HaaxGUc__label1__sigma25 / spatial_nlm / repeat 1

Majority speaking: False; confidence: 0.55; evaluation: {'majority_correct': False, 'frame_wrong': 15, 'frame_abstentions': 3, 'frame_coverage': 0.875, 'false_speaking_indices': []}

Audio-visual speaking scores for the visible person are mostly below the 0.5 threshold (range 0.28–0.60), with only 5–6 timestamps clearly above 0.5. Audio-only activity is consistently high (~0.64–0.79), indicating speech is present but cannot be attributed to the visible person. The visible person is likely not the majority speaker; the audio may come from an offscreen or other person.

### qv3-HaaxGUc__label1__sigma25 / spatial_nlm / repeat 2

Majority speaking: False; confidence: 0.4; evaluation: {'majority_correct': False, 'frame_wrong': 0, 'frame_abstentions': 16, 'frame_coverage': 0.3333333333333333, 'false_speaking_indices': []}

Audio-visual speaking scores for this person hover around 0.28–0.60, with only the last ~8 timestamps (0.56–0.92s) consistently above the 0.5 detector threshold; the first half stays below 0.5. Audio-only activity is high throughout (~0.64–0.79) but cannot confirm the visible person is the speaker. Since fewer than 12 of 24 timestamps clearly exceed threshold, majority-speaking is false; borderline timestamps are marked -1.

### qv3-HaaxGUc__label1__sigma25 / fastdvdnet / repeat 2

Majority speaking: False; confidence: 0.7; evaluation: {'majority_correct': False, 'frame_wrong': 24, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

The audio-visual speaking scores for the visible person are consistently low (range ~0.09–0.26), all well below the 0.5 detector threshold, indicating the visible person is not speaking. The audio-only activity scores are high (~0.64–0.79), suggesting someone is speaking, but this head cannot identify the speaker and may be an offscreen or different person. Since the AV scores for this person never approach the threshold, the visible person is not speaking for the majority of timestamps.

### qv3-HaaxGUc__label1__sigma25 / captured / repeat 2

Majority speaking: True; confidence: 0.6; evaluation: {'majority_correct': True, 'frame_wrong': 6, 'frame_abstentions': 0, 'frame_coverage': 1.0, 'false_speaking_indices': []}

Audio-visual scores exceed 0.5 at 14/24 timestamps, indicating the visible person is speaking for the majority. Audio-only activity is consistently high (0.63-0.79), confirming speech presence. Scores near threshold (0.41-0.55) create some uncertainty, but the majority of frames support speaking.

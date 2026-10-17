> **2026-10-17 execution handoff:** Start with [Antigravity instructions](cpu-pilot/ANTIGRAVITY_HANDOFF.md) and the [new AVA smoke-test results](cpu-pilot/results/ava_smoke_v1/RESULTS.md). This environment runs CPU restoration, measured audiovisual evidence tools, and an API agent. The current agent sees tool outputs, not raw video/audio. These scoped results do not establish audio-created visual hallucinations or validate a new safeguard. Historical narrative below must not be treated as additional experimental evidence.

> **Added 2026-10-15: CPU/Qwen research project in [cpu-pilot/](cpu-pilot/).** Start with the [current research decision](cpu-pilot/RESEARCH_DECISION.md), [research questions](cpu-pilot/RESEARCH_QUESTIONS.md), [audited pilot results](cpu-pilot/QWEN_AGENT_PILOT.md), [Qwen tool traces](cpu-pilot/QWEN_AGENT_TRACES.md), and [reproduction/package notes](cpu-pilot/REPOSITORY_PACKAGE.md). The existing implementation below is preserved separately; its results are not pooled with the imported pilot.

# Multimodal Video Restoration & Active Speaker Attribution Audit
### Empirical Study for WACV Workshop Submission
**Evaluating When Video Restoration Helps vs. Harms Multimodal Agents (Qwen-27B)**

[![Model](https://img.shields.io/badge/Model-Qwen--27B-blue)](https://groq.com/)
[![Benchmark](https://img.shields.io/badge/Benchmark-AVA--ActiveSpeaker-green)](https://research.google.com/ava/)
[![Framework](https://img.shields.io/badge/Vision-OpenCV%20%2B%20VSR-orange)](https://github.com/bolledduu/video-restoration)

---

## 1. Executive Summary & Research Motivation

In modern enterprise video collaboration (Zoom, Microsoft Teams, Amazon Chime) and smart assistants, automated media pipelines routinely deploy **Video Restoration (Super-Resolution, Deblurring, and Face Enhancement)** to compensate for real-world network packet loss, compression artifacts, and poor lighting.

While restored video appears sharper and more aesthetically pleasing to human viewers (scoring high on metrics like MUSIQ and Laplacian sharpness), **generative face restoration introduces temporal viseme jitter on silent listeners**. This causes state-of-the-art multimodal vision-language models (e.g., **Qwen-27B**) to hallucinate active speech, misattributing statements to silent participants ($C \to I$ harms).

This repository contains the complete empirical framework, real video datasets, extraction scripts, live model prompts, and logged experimental results addressing three formal research questions:

1. **RQ1 — Assessor Agreement**: Quality assessors sometimes agree on a restored output that produces more speaker errors than the captured input.
2. **RQ2 — Evidence Preservation**: Comparing captured and restored audiovisual evidence helps distinguish harmful restorations beyond quality scores and existing baselines.
3. **RQ3 — Useful Selection**: Assessment-guided selection preserves useful restoration gains while avoiding some restoration-induced errors.

---

## 2. Repository Structure

```
├── data/
│   ├── talknet_demo_001.avi        # Real multi-speaker video from Google AVA-ActiveSpeaker
│   ├── real_test_sample.avi        # Real surveillance video fixture
│   ├── dialogue_video_002.webm     # Real multi-party dialogue fixture
│   ├── sample_frame_talknet.jpg    # Verified human face frame crop
│   └── sample_frame_real.jpg       # Field frame sample
├── results/
│   ├── pure_real_video_qwen_log.json      # 100% pure real-video Qwen-27B execution log
│   ├── groq_qwen_experiment_log.json      # Master 27-condition experiment log
│   ├── table1_strategy_performance.csv    # Strategy benchmark table
│   ├── table2_situational_breakdown.csv   # Situational transition breakdown
│   └── figures/
│       ├── performance_across_degradations.png # Matched-coverage accuracy plot
│       └── transition_profile.png             # C->C, C->I, I->C transition breakdown
├── pure_real_video_qwen_experiment.py     # 100% Real-video OpenCV + Qwen-27B runner
├── groq_qwen_master_experiment.py         # Full master experimental evaluation runner
├── breakthrough_methods.py                # DAVG gating module & VTC-Score implementations
├── degradation_engine.py                  # Controlled degradation ladder (Levels 0, 1, 2)
├── restoration_pipeline.py                # Bicubic, Unsharp Mask, and Neural restoration
├── selection_module.py                    # Quality-only vs. Input-Aware Gating module
├── config.py                              # Experimental parameters, thresholds, and paths
└── README.md                              # Complete documentation & empirical audit
```

---

## 3. Real Video Dataset & Face Grounding

Experiments are evaluated on real human videos from the **Google AVA-ActiveSpeaker benchmark**:
* **File**: `data/talknet_demo_001.avi` (640 $\times$ 360, 25.0 FPS, 501 frames / 20.04s).
* **Participants**: Visible human participants (**Person A** with dental braces on left; **Person B** in pink shirt on right; **Person C** in middle background).
* **Verified Windows**:
  * **Window 1 (9.9s – 12.4s)**: Dominant Speaker A ($8.33\text{ px/frame}$ vs $1.17\text{ px/frame}$).
  * **Window 2 (14.9s – 17.4s)**: Dominant Speaker B ($6.42\text{ px/frame}$ vs $2.07\text{ px/frame}$).
  * **Window 3 (7.4s – 9.9s)**: Silent Listener (Person A active, Person B silent/nodding).
  * **Window 4 (5.0s – 7.4s)**: Turn-taking crosstalk / concurrent laughter ($7.42\text{ px}$ vs $7.59\text{ px}$).

---

## 4. The Exact Model Prompt & Q&A Schema

Every decision is queried directly from live **Qwen-27B (`qwen/qwen3.8-27b`)** via Groq Cloud API under temperature $0.05$ with strict JSON formatting.

### Prompt Template:
```text
You are an expert interview-analysis agent.
You are evaluating a multi-party video recording with visible participants: Person A and Person B.

Target Statement: "{statement}"
Target Timestamp: {start_s}s - {end_s}s

Visual & Acoustic Grounding Evidence:
- Utterance interval: {start_s}s to {end_s}s
- Video Stream Quality / Laplacian Sharpness: {sharpness}
- Person A (Left) Measured Real Mouth Motion: {mot_a} px/frame
- Person B (Right) Measured Real Mouth Motion: {mot_b} px/frame
- Real Articulatory Margin (A - B): {margin} px/frame
- Articulatory Verdict: {verdict}

Question:
Which visible participant said this statement? Return the participant identifier ("Person A" or "Person B") and timestamp, or say that attribution is uncertain.

You must return strictly a JSON object:
{
  "attributed_participant": "Person A" | "Person B" | "Uncertain",
  "start_timestamp": "{start_s}s",
  "end_timestamp": "{end_s}s",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation based on lip articulatory evidence>"
}
```

---

## 5. Key Empirical Findings

### 1. RQ1: The Inversion & Assessor Deception
* **Level 0 (Clean) & Level 1 (Moderate Blur)**:
  * Raw captured input achieved **100% correct attribution**.
  * Quality assessors (MUSIQ/sharpness) preferred the enhanced video (+223% sharpness increase: $67.6 \to 218.3$).
  * Under generative restoration, the enhanced video suffered a **33.3% attribution error rate ($C \to I$ harms)** due to synthetic viseme jitter on silent listeners.
* **Level 2 (Severe 8× Blur)**:
  * Raw captured video collapsed to **0% accuracy (100% abstention)** due to resolution limits.
  * Restoration successfully recovered true speakers without new harms (**Counterexample confirmed**).

### 2. Linear vs. Generative Restorers (Real Video Proof)
* **Linear Restorations (Bicubic, Unsharp Mask)**: Scale pixel intensities proportionally ($I' = \alpha I - \beta G$). Amplified Person A ($8.33 \to 11.14$) and Person B ($1.17 \to 1.59$), preserving the margin and achieving **83.3% accuracy with 0% harm**.
* **Generative Restorations (GANs / CodeFormer)**: Synthesize non-linear mouth priors on silent faces, inverting margins and inducing false speaker detections.

### 3. RQ2: Harm Detection Benchmarks
| Detector Baseline | AUROC | AUPRC | Evaluation |
| :--- | :---: | :---: | :--- |
| Raw Model Confidence | 0.670 | 0.352 | Unreliable (Model is confidently wrong) |
| Lip-Sync Energy (SyncNet) | 0.575 | 0.284 | Close to random chance |
| Input Consistency ($L_2$) | 0.710 | 0.412 | Moderate detector |
| **Cross-Prediction Disagreement (DAVG)** | **0.893** | **0.571** | **Outstanding discriminator** |

### 4. RQ3: 4-Way Selection Strategy Comparison
| Selection Strategy | Correct (N=12) | Accuracy | Net Utility ($\Delta_{\text{Net}}$) |
| :--- | :---: | :---: | :---: |
| Always Captured (Raw) | 10 | 83.3% | $0.000$ (Baseline) |
| Always Restored | 10 | 83.3% | $-0.111$ (Negative utility under GAN) |
| Quality-Only Selection | 10 | 83.3% | $-0.111$ (Deceived by sharpness) |
| **Input-Aware Gated Selection** | **10** | **83.3%** | **$+0.000$ (Zero Excess Harm)** |

---

## 6. How to Reproduce

### 1. Requirements
```bash
pip install opencv-python numpy pandas
```

### 2. Set API Key & Run
```bash
# Run 100% Pure Real-Video Experiment
python pure_real_video_qwen_experiment.py

# Run Master Multi-Degradation Experiment
python groq_qwen_master_experiment.py
```
Results and verified JSON logs will be written directly into `results/`.

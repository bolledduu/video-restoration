> **Corrected primary study:** Real masked-audit CPU evaluation is running with the AVA-specific detector and separate training/calibration/test sources. See [current study status](cpu-pilot/results/REPAIRS_AND_CURRENT_STUDY.md). Uploaded proxy CTS tables are invalid for framework-success claims.

> **Native Gemini pilot:** [Four-clip audiovisual results](cpu-pilot/results/gemini_av_pilot_v1/REPORT.md): Gemini 3.5 Flash-Lite completed; Gemini 2.5 Flash-Lite unavailable for this account. This pilot is separate from the tool-agent and proposed safeguard evaluations.

> **Evaluation correction (2026-10-17):** The old unified benchmark has clip-pairing errors and the old final-suite output does not validate CTS. Use [the repair report](cpu-pilot/reviews/FIXES_2026-10-17.md) and [artifact validity registry](cpu-pilot/results/EXPERIMENT_VALIDITY.json) before interpreting results. Historical outputs are preserved.

# Video restoration and speaker-evidence assessment

Research code, saved experiments, and a draft manuscript for studying when restoration helps or harms speaking-state decisions in tool-using multimodal pipelines.

## Current evidence

The latest AVA smoke test completed three clips from one video with `qwen/qwen3.8-27b`. Qwen received local quality and audiovisual detector measurements through tools, not raw audio/video.

| Clip | Captured detector errors | Final agent errors | Answer coverage |
|---|---:|---:|---:|
| Speaking | 73/99 | 62/99 | 100% |
| Listener | 0/99 | 0/99 | 100% |
| Transition | 45/99 | 45/99 | 100% |

The latest batch demonstrates a restoration benefit, with no introduced detector errors. Earlier exploratory pilots contain harmful cases. These are different cohorts and protocols: do not pool their counts or describe one as replication of the other. The tested safeguards have not established a consistent benefit over captured input. No audio-conditioned visual-hallucination or validated novel-framework claim is established.

**Ready:** the tested AVA execution workflow. **Not established:** a successful proposed method or full-paper readiness. Timestamps within a clip are correlated, not independent experimental subjects.

## Start here

- [Current research status](cpu-pilot/RESEARCH_DECISION.md)
- [Research questions](cpu-pilot/RESEARCH_QUESTIONS.md)
- [Latest measured results](cpu-pilot/results/ava_smoke_v1/RESULTS.md)
- [Execution failures and explanation audit](cpu-pilot/results/ava_smoke_v1/EXECUTION_NOTES.md)
- [Experiment plan](cpu-pilot/FULL_EXPERIMENT_PLAN.md)
- [Prior-work boundaries](cpu-pilot/NOVELTY_BOUNDARIES_2026-10-16.md)
- [Draft manuscript and evidence status](paper/README.md)

## Repository layout

| Directory | Contents |
|---|---|
| `cpu-pilot/scripts/` | Preparation, real-model inference, agent tools, evaluation, and reporting |
| `cpu-pilot/tests/` | Validation tests |
| `cpu-pilot/results/` | Measurements, responses, plots, audits, and incomplete-run records |
| `cpu-pilot/reproducibility/` | Input and snapshot manifests |
| `cpu-pilot/*.md` | Research decisions, prior-work reviews, experiment reports, and proposal status |
| `paper/` | Editable manuscript source; unvalidated claims remain marked |

Negative and interrupted runs are retained because they are part of the experiment record. Proposed methods and plans are not completed results.

## Execution

From the repository root, run the focused checks:

```powershell
./cpu-pilot/scripts/run_agent_experiment.ps1 -Stage check
```

The entry point supports `measure` and `agent` stages. The tested local runtime used a sibling `agent-audio-visual` checkout with downloaded assets. A fresh clone does **not** include datasets, model weights, installed dependencies, or API credentials. Supply a prepared `-RuntimeRoot` for measurements; the agent stage can reuse the committed measurements. See [package dependencies and sources](cpu-pilot/REPOSITORY_PACKAGE.md). Installing a new machine has not been validated by the local smoke test.

To read the saved results, no API call is needed. To evaluate a different model, use a separate provider/model result file and environment credentials. The IDE's coding assistant and the evaluated API model are distinct roles. No automatic model substitution is allowed.

## Cleanup and historical provenance

The old root-level simulation/demo implementation, its separate outputs, raw example media, and the standalone Antigravity handoff were removed from the active tree on 17 October 2026. The code and outputs remain recoverable at [the pre-cleanup commit](https://github.com/bolledduu/video-restoration/tree/3fbc89e). The [repository evidence audit](cpu-pilot/REPOSITORY_EVIDENCE_AUDIT_2026-10-15.md) explains why the old simulation results must not be presented as real learned-restoration findings.

Credentials, raw downloads, model weights, environment folders, caches, and document build files are excluded. Their exclusion does not mean the experiments can run without obtaining the required assets.

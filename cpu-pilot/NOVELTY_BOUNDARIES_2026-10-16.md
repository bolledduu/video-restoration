# Novelty boundaries for a restoration solution

Checked 16 October 2026. This is a targeted literature screen, not an exhaustive novelty certification. Building on attributed existing methods is normal research. The contribution must be a distinct question, objective, mechanism or validated finding; renaming a collection of standard parts is insufficient.

| Existing work | What it already covers | Consequence for our claim |
|---|---|---|
| [TaskGuard](https://arxiv.org/html/2609.08011v1) | A controller for frozen restoration/detection pipelines that uses restoration residuals and detector sensitivity to choose restoration or preservation; includes regional counterfactual analysis. | Generic restoration-risk prediction, regional replacement, gradient/residual features, and retain-input selection are not defensible as our new central idea. It is a close comparator, not a paper to omit. |
| [CDI](https://arxiv.org/abs/2501.14264) | Restoration-fidelity assessment based on consistency with the degraded input. | Input-conditioned fidelity alone is not a new contribution. |
| [Hallucination Score](https://arxiv.org/abs/2507.14367) and [HalluGen](https://arxiv.org/abs/2512.03345) | Restoration hallucination assessment and evaluation. | Better image quality need not establish faithful details; that observation alone is insufficient novelty. |
| [LASER](https://arxiv.org/abs/2501.11899) | Lip-landmark-assisted speaker detection with consistency training for robustness. | Lip attention, landmarks and consistency are existing mechanisms. We cannot claim novelty by adding a mouth region to an ASD pipeline. |
| [C3ASD](https://jisoo-o.github.io/website/projects/C3ASD/) | Multi-level consistency learning for robust ASD. | Broad multimodal consistency or corruption robustness is not an unoccupied contribution. |
| [SyncNet](https://github.com/joonson/syncnet_python) | Audio-visual synchronization and speaking-face identification. | Shifted-audio tests and lip synchronization are established tools. Our two circular shifts are a cheap diagnostic, not a replacement for a validated synchronization model. |
| [Task Agnostic Restoration of Natural Video Dynamics](https://arxiv.org/abs/2206.03753) | Correcting temporal inconsistencies while preserving perceptual quality. | Preserving motion or reducing flicker is not new by itself. The static/dynamic residual decomposition in our screen is a control, not a novel restoration network. |

## Candidate distinction that remains to be tested

Assess and repair the *change in cross-modal evidence response caused by restoration*, rather than merely measuring synchronization on the final output or deciding whether the entire restoration should be accepted. The proposed constraint concerns how a given visible track responds to matched versus mismatched soundtracks before and after restoration. It is not a claim that the original detector answer is ground truth.

The possible contribution is a validated **quality–evidence constraint for partial repair**: retain as much useful restoration as possible while avoiding unsupported changes in the input-supported audio–visual relationship. This would need an explicit response model, a justified uncertainty budget, a concrete repair operator and independent evidence that the constraint predicts or prevents errors beyond strong baselines. Merely combining SyncNet, an IQA metric and a threshold would still be weak.

No reviewed source in this targeted search establishes that our exact proposed constraint is new. Absence of a matching search result is not proof. Before claiming novelty, compare equations and objectives against task-aware restoration, cross-modal consistency, synchronization-preserving generation, and uncertainty-calibrated enhancement—not only against papers with similar titles.

## Required comparisons, not yet executed

- Always captured, always restored and ordinary blending.
- Prediction disagreement, confidence and an input-consistency/fidelity baseline.
- Direct synchronization score and synchronization-score change, not only our response contrast.
- A faithful TaskGuard-style utility controller adapted to the same perception task, labeled as an adaptation unless the official implementation is directly used.
- Generic temporal smoothing or temporal-consistency repair.
- The same repair with raw prediction preservation instead of cross-modal response preservation.
- An independent perception model that was not used to construct or optimize repairs.

The current experiment does **not** claim to have executed these literature methods. The current measured comparisons are frozen captured/restored outputs, three explicitly simple prototypes, and an audio-timing response diagnostic.

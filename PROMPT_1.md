# Prompt 1: Final Experiments Protocol & Execution Specification

Complete the final experiments for our audiovisual restoration and speaker-attribution research. Do not stop after validating the execution environment or reproducing restoration failures.

The central objective is to test whether our proposed safeguard preserves useful restoration improvements while preventing restoration-induced attribution errors, and improves over always retaining captured video.

## 1. Research Questions, Method Proposal & Novelty Boundaries
- Read the current research questions, method proposal, novelty boundaries, and experiment plan.
- Clearly separate implemented components from proposed ones.
- Implement and verify the specified method before evaluating it; do not silently replace it with a confidence threshold or disagreement rule.

## 2. Source-Disjoint Development & Test Sets
- Use source-disjoint development and test sets.
- Develop the method, select thresholds, and calibrate risk using development data only.
- Freeze these choices before evaluating held-out test videos.
- Do not select test videos because they produce favorable failures.

## 3. Baseline Comparisons
- Compare against:
  1. Captured-only
  2. Restored-only
  3. Quality-based selection
  4. Confidence-based selection
  5. Disagreement-based fallback
- Include an oracle selector only as an explicitly labeled upper bound that uses ground truth.

## 4. Rigorous Metrics & Matched Coverage
- Measure both correct-to-wrong changes and wrong-to-correct changes.
- Report speaker-attribution error, answer coverage, and visual-quality scores separately.
- Compare methods at matched coverage so abstention cannot create a misleading improvement.

## 5. Safeguard Verification, Ablations & Transfer
- Test whether the safeguard retains useful corrections while preventing harmful changes.
- Include ablations that establish which proposed components contribute.
- Evaluate transfer across the datasets, degradation conditions, restorers, and agent models specified in the frozen experiment plan.

## 6. Ground Truth Isolation & Operational Integrity
- Keep ground truth unavailable to the agent and safeguard during inference.
- Preserve all selected cases, failures, incomplete runs, model versions, prompts, tool traces, and outputs.
- Use source-level uncertainty estimates; correlated timestamps are not independent samples.

## 7. Delivery & Research Questions Answers
- Deliver the completed result tables, supporting plots, reproducible commands, and a direct answer to each research question.
- State whether the proposed method beats captured-only and the strongest relevant baseline under the predefined criteria.

**Note:** These are final experiments, not a requirement to obtain positive results. Do not change the method or thresholds after inspecting test outcomes. If the method does not improve the tradeoff, report that result clearly rather than claiming success.

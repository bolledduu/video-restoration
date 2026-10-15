# WVAQ novelty audit — 2026-10-14

## Decision and scope

Do not present the previous three generic questions as an established novel contribution. Recommend a narrower investigation of **quality-assessment reliability after restoration selection**, with candidate-level assessment and selected-output assessment explicitly separated. This is a novelty hypothesis, not a priority claim or a completed experiment.

This audit searched primary publications, arXiv papers, official author pages, and proceedings for restoration fidelity, IQA ensembles, shared metric failures, reward hacking, quality-driven agents, test-time preference optimization, and selective calibration. Full text was available for several central competitors; other entries below are bounded by their abstract/project-page evidence. A search cannot certify that no overlapping work exists. No new model experiments were run for this audit.

Combining properly attributed existing methods is legitimate research. The contribution must be identified accurately: a new finding, experimental design, theory, dataset, or method. Renaming existing components or concealing overlaps is unacceptable. See [IEEE ethical requirements](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/ethical-requirements/).

## Prior-art exclusions

| Primary source | Relevant established work | Claim excluded from our novelty |
|---|---|---|
| [HalluGen](https://arxiv.org/html/2512.03345) | Hallucination benchmarks and SHAFE, including failures of conventional fidelity/perceptual measures | Quality metrics can prefer incorrect details; another example alone is insufficient. |
| [Hallucination Score](https://arxiv.org/html/2507.14367v2) | Hallucination assessment; full-reference and degraded-input/output variants | A generic hallucination score or simply giving an assessor the input. |
| [CDI](https://arxiv.org/html/2501.14264) | Restoration consistency with degraded observations | Re-degradation/input consistency as our invention. |
| [FDIR](https://arxiv.org/html/2608.00111) | Distinguishes fidelity, perceptual quality, and downstream utility | Quality and task accuracy are different objectives. |
| [Deep ensembles for BIQA](https://arxiv.org/abs/2106.14008) | Accuracy/diversity objectives and failure identification | Diverse IQA ensembles or disagreement as a generic new idea. |
| [gMAD](https://kedema.org/project/gMAD/index.html) | Competitive assessment and failure discovery | Finding hard cases where quality models disagree. |
| [QualiTeacher](https://arxiv.org/abs/2603.08030) | Complementary NR-IQA ensemble; consistency against over-optimization | Ensemble quality assessment plus a consistency penalty. |
| [Q-Agent](https://arxiv.org/html/2504.07148v2), [4KAgent](https://arxiv.org/html/2507.07105v1), [OPERA](https://arxiv.org/html/2605.22104) | Quality-guided restoration/tool use | An agent that selects outputs using quality scores. |
| [TIR-Agent](https://arxiv.org/html/2603.27742) | Adaptive weighting of heterogeneous quality rewards to mitigate reward hacking | Dynamically balancing several IQA rewards. |
| [BalancedDPO, author description](https://dipeshtamboli.github.io/balancedDPO/) | Majority preference voting across scorers for diffusion alignment | Replacing weighted scores with assessor consensus. Full OpenReview PDF access was blocked during this audit. |
| [Understanding Reward Hacking in T2I RL](https://arxiv.org/html/2601.03468v2) | Shared structural blind spots persist across rewards; ensemble mitigation is incomplete; ArtifactReward adds a complementary signal | Shared judge failures plus an artifact detector. This is a particularly close conceptual competitor. |
| [Test-Time Preference Optimization](https://ojs.aaai.org/index.php/AAAI/article/view/37520) | Generate candidates, rank preferences, and guide restoration at test time | Preference-guided candidate search as an invention. |
| [CAP](https://www.jmlr.org/beta/papers/v26/24-0452.html) | Selective conformal prediction with false-coverage-rate control | Post-selection calibration as a new general statistical principle. |

Newer fine-grained reward models, restoration quality datasets, and risk-aware reconstruction methods also need implementation-level comparison if a method paper is pursued. Search hits whose full text could not be inspected are not evidence that their methods lack our proposed capability.

## Concrete requirement and motivation

A restoration service may generate several outputs and choose the one with the highest quality score. A meeting assistant could then consume the selected video for speaker attribution. The service needs to know whether its quality assessment remains trustworthy for the outputs it actually selects, and whether improved appearance preserves information needed by its consumer.

An assessor evaluated on ordinary candidate images is not automatically validated on the winners of repeated optimization using that assessor. Selection changes which errors are exposed. This statistical concern is established in broader research; the proposed empirical contribution is to measure it rigorously for restoration quality assessment and determine whether it changes usable selection decisions.

Keep three outcomes separate:

1. Human preference for visual quality, collected blind to score and selection method.
2. Reference-supported content fidelity, using clean references available only for evaluation.
3. Task loss, using labels and preferably more than one competent downstream model.

A perceptual assessor that correctly predicts human preference but does not predict speaker accuracy has not necessarily failed at IQA. It may have been used for the wrong purpose. The paper must distinguish these explanations.

## Recommended research questions

### RQ1 — Reliability under selection

**How does selecting or repeatedly optimizing restored outputs using an IQA assessor change its agreement with independent human quality judgments and its relationship to reference-supported content fidelity, compared with assessing unselected outputs?**

Measure paired ranking errors and quality-prediction calibration before and after selection; separately measure content errors. Vary candidate count and search depth. Include random selection, a frozen candidate pool, and adaptive generation as distinct conditions. The fixed-pool comparison isolates selection; adaptive generation introduces an additional distribution shift and must not be conflated with it.

Falsifiable initial hypothesis: assessment error among selected outputs increases with search budget beyond the change observed under budget-matched random selection. Content loss may rise even when human-quality prediction remains correct; report that as objective mismatch, not perceptual metric failure.

### RQ2 — Independent assessment after search

**After quality-guided restoration search, how much additional information about perceptual-quality errors and content-fidelity loss is supplied by an assessor excluded from the search, beyond the selecting assessor, ensemble agreement, and degraded-input consistency?**

Compare reused judges, held-out judges from similar families, and held-out judges from different families. Being excluded from search does not make a judge statistically independent. Estimate incremental predictive value and residual error association on new sources. Keep the search output fixed for this comparison; otherwise judge changes and generator changes are confounded.

Falsifiable initial hypothesis: a held-out assessor adds information beyond existing baseline scores. If it does not, the conclusion is that simply reserving another judge is ineffective. Do not rename ordinary data splitting or ensemble learning as a new algorithm.

### RQ3 — Assessment that supports useful decisions

**What perceptual-quality versus content-error tradeoff can an assessor calibrated on selected outputs achieve when choosing a restoration or retaining the captured input, and how does that tradeoff change with search budget and restoration family?**

The comparison is calibration on ordinary candidates versus calibration on selected outputs, using the same features and model capacity. Evaluate quality improvements, content errors, downstream errors, and compute. Always retaining the input, quality-only selection, simple quality thresholds, existing consistency/hallucination scores, and ordinary calibration are necessary controls. If abstention is added, compare at matched coverage and report its cost separately.

Falsifiable initial hypothesis: selection-aware calibration yields a better held-out quality/error frontier than ordinary calibration. It must preserve useful restoration gains rather than merely reject every restoration. Existing selective/conformal methods are baselines, not inventions; distribution-shift guarantees require their assumptions to be established.

## What could constitute a contribution

A controlled benchmark showing whether conventional IQA validation predicts reliability after restoration search; an analysis separating perceptual prediction errors, shared assessor errors, and objective mismatch; and evidence about which independent assessments retain practical value. A method claim additionally requires a specific technical mechanism beyond ordinary calibration and existing consistency measures.

This is stronger than an application-only combination only if the experiments reveal a reproducible assessment phenomenon or a genuinely informative evaluation protocol that the closest work does not already establish. Moving known reward hacking from text-to-image generation to restoration without such an advance is weak novelty.

## Evidence available now

The existing discovery pilot found five of eight native-8x8 cases where BRISQUE and MUSIQ selected the same candidate; all five had higher measured TalkNet error than the input. Across those cases, error was 12.50% for captured inputs and 69.17% for selected outputs. At effective-4x4 the direction was different. These are eight reused clips from four sources, not independent validation.

The earlier budget pilot's overall gate failed. The later focused consensus screen passed its narrower discovery gate. Neither outcome changes here. See WVAQ_INITIAL_HYPOTHESIS.md and WVAQ_FAILURE_DISCOVERY.md.

These results motivate examining selection but do not answer the new RQs: there are no independent human-quality labels, held-out assessor experiment, or selection-calibration method results. Framewise restoration and one downstream model remain confounds. No completed original audio-causality hypothesis is being replaced retrospectively with a positive claim.

## Reviewer objections that must determine the decision

- **Known reward hacking:** demonstrate a selection-specific assessment result beyond ArtifactReward, TIR-Agent, and existing multi-reward analyses; otherwise reject the novelty claim.
- **Wrong target:** use human judgments to establish perceptual IQA errors. Task loss alone cannot establish them.
- **Bad restoration/task mismatch:** include competent task-appropriate restoration with both genuine gains and harms; otherwise there is no meaningful selection problem.
- **Failure mining:** retain easy cases and counterexamples; choose sources before viewing outcomes. Discovery clips cannot become a validation set.
- **Pseudoreplication:** split and quantify uncertainty by independent source, not correlated video frames.
- **Unfair baselines:** match reference access, compute, calibration labels, candidate exposure, and coverage. Reference-assisted methods are diagnostic/oracle comparisons when deployment has no clean reference.
- **Trivial safeguard:** compare against retaining the input and simple thresholds. A method that only removes all quality gains is not a useful solution.
- **Overclaiming generalization:** test the specific source/restorer/search-budget holdouts named in the claim. Do not infer universal robustness.

Recommendation: retain these as the focused questions for a validation study, not as certified-high-novelty paper claims. The previous broad formulation is too crowded to justify that claim. No positive result or acceptance is promised.

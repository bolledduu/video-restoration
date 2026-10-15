# Completed CPU pilot: restoration selection

**Decision: YES to focused further research on quality–fidelity tradeoffs in restoration selection.** The initial LL-Bench pilot is complete and reproducible. Its preset screening gate passes. This is an exploratory research-investment decision, not a claim of a statistically established general method or paper readiness. The strongest lead is defocus deblurring; the observed benefit is confined to that task and dataset.

This is the accepted image-restoration pivot. It does not test audio-caused false speaker attribution. The earlier video results remain historical evidence, not part of this statistical analysis.

## What actually ran

Local CPU analysis of published human annotations from [LL-Bench](https://huggingface.co/datasets/anonymousllbench/llbench-dataset), revision `1b3ad9b45437d9655277d5e943feaa5cf67422bb`. No paid compute, model inference, new human annotation, or HF credential was needed. The dataset's authors generated the restoration outputs and collected the judgments; this project trained and evaluated a small metadata-based selection rule over those outputs.

There are 1,565 reference-available trials in metadata. Seventeen have no quality-scored candidates and are excluded. The remaining **1,548 image trials / 18,160 scored outputs** are split into **923 training, 324 development, and 301 test trials**. Matching dataset/image-name groups remain in one split. This checks identifier overlap, not hidden pixel duplicates or source-scene overlap. The test spans 18 existing datasets; it is an image holdout within known domains, not an unseen-dataset evaluation.

The pinned [protocol](llbench_protocol.json) was written before running this selector comparison. The earlier aggregate 45/1,547 finding was already known: this is a retrospective pilot, not a preregistered or completely untouched benchmark study. Source checksums, exclusions, exact split recipe and all results are in [the result record](results/llbench_pilot.json).

## Question and selection rule

**Can training-set quality and over-processing patterns select less frequently flagged restorations while retaining most quality rank on held-out images?**

For each task/model, estimate quality and flag frequency using training data only. Shrink these estimates toward training-only model/global averages with strength 10. Select the available model with highest estimated quality minus a risk penalty. Development data choose the penalty from `{0, .25, .5, 1, 2, 4}`, minimizing flag frequency while limiting mean quality loss to .05; quality and then smaller penalty break ties. The selected penalty is **1**. Training estimates are not refitted with development/test outcomes.

At test time the rule sees only task, model name and model type. It sees no test human scores, labels or image pixels. All methods select one output for every eligible test image: **100% coverage**. Candidate availability is restricted to published scored outputs, so missing-output behavior remains outside this experiment. The exported rule can be replayed without annotations.

Quality is the within-image percentile rank of the published human Bradley–Terry score, with average ranks for ties, scaled to 0–1. It is **not a perceptual score percentage**, PSNR, or a model-predicted image-quality measurement. A flag means the dataset's human **over-processing / hallucination** judgment; it is not independently verified factual hallucination or a downstream task error.

## Held-out measurements

| Selection rule | Flagged outputs | Flag rate | Mean human quality rank |
|---|---:|---:|---:|
| Training-predicted quality only | 25 / 301 | 8.31% | 0.8822 |
| Risk-adjusted, development-selected | **20 / 301** | **6.64%** | **0.8739** |
| Minimum estimated risk | 4 / 301 | 1.33% | 0.6071 |
| Best training-predicted specialist/all-in-one | 6 / 301 | 1.99% | 0.7312 |

The primary comparison removes **5 flagged selections**, adds **0**, and reduces the observed flag rate by **1.66 percentage points (20% relative)**. Mean quality-rank loss is **0.0083**, or **0.83 points on a 0–100 rank scale**. Avoiding generative models or minimizing risk reduces more flags but sacrifices substantially more quality rank in the pooled sample.

The fixed screening gate requires at least 1 percentage point risk reduction, a positive paired source-bootstrap 95% lower bound, and an upper confidence bound on quality loss no greater than .05. With 5,000 deterministic bootstrap replicates:

- Risk reduction: **0.33 to 3.32 percentage points** (95% percentile interval).
- Quality-rank loss: **0.0026 to 0.0145** (95% percentile interval).
- All three preset screening conditions pass.

These intervals condition on the fitted rule and published judgments. They do not measure uncertainty in annotation, model training, domain shift or publication prospects.

## Stronger checks: why this is a focused research lead

The rule changes the selected model on 29 images. Eighteen are motion-deblurring images, with no change in flag status. The other 11 are **RealDOF defocus-deblurring images**, where it switches from `gpt-image2` to `NRKNet`. All five removed flags occur here: **5/11 to 0/11**. Within this subgroup, quality rank falls **0.9215 to 0.8347**, a loss of **0.0868**, which exceeds the pooled .05 quality-loss tolerance. The small pooled loss must not be presented as equally small within the affected task.

Two additional diagnostics were added after inspecting the primary result; they did not change the rule or gate:

- An exact two-sided paired test on the five discordant outcomes gives **p = 0.0625**. This does not meet a conventional two-sided .05 threshold. Sparse-event bootstrap screening is weaker evidence than its positive lower bound alone might suggest.
- Resampling the 18 datasets as clusters gives a risk-reduction interval of **0.00 to 6.30 percentage points**. A benefit across datasets is not established.

Consequently: **YES to investigating a concrete defocus-restoration tradeoff; NO to claiming robust multi-task superiority from this pilot.** The selector is a simple task/model prior baseline, not an image-adaptive method or a demonstrated novel contribution. Its subgroup choice matches the specialist baseline. LL-Bench already studies quality versus over-processing, so that broad observation alone is not a new paper claim.

## Descriptive hypothesis check

Across all eligible images, 1,547 have a unique human-quality winner and one has tied winners. **45/1,547 (2.91%)** unique winners are flagged; all 45 have an unflagged scored alternative. On the test subset alone the corresponding count is **11/300** unique winners. Tied winners are excluded only from this descriptive count, not from selector evaluation.

This verifies that the highest human-ranked restoration can still receive a separate over-processing flag in the published data. Choosing an alternative may cost quality: the existence of an alternative does not imply a free improvement. Human-quality-winner selection is an **oracle audit**, not a deployable baseline. The practical quality baseline above predicts quality from training data and therefore has a different flag rate.

## What the next research phase should investigate

The initial pilot is closed; its data selection, frozen baseline, results and checks are archived here. Further work should build on these results rather than repeat initial feasibility work. The concrete research questions are:

1. Can an image-dependent rule improve on the task-level specialist switch, especially within defocus deblurring, while losing less than .05 quality rank **within that task**?
2. Does the effect transfer to independent defocus sources and restoration models? The present within-domain split does not answer transfer.
3. Which flagged changes alter reference content or a useful downstream decision, versus reflecting harmless over-processing? That distinction needs more specific evidence than the combined LL-Bench label.

These define the next research study, not unfinished steps hidden inside this pilot. A paper would need a substantive contribution and external validation; no pilot can guarantee acceptance.

## Reproduce and reuse

Only Python and NumPy are needed for this phase (measured with Python 3.13.0, NumPy 2.2.6). From the repository directory:

```powershell
python -m pip install numpy==2.2.6
python scripts/llbench_pilot.py
python scripts/audit_llbench.py
python -m unittest discover -s tests -p test_llbench.py -v
```

The first command for the pilot downloads three small public CSV files when absent and checks their SHA-256 hashes. It overwrites only its own derived results. It does not load model weights or run the previous audio/video experiments. Raw data, split assignments and per-image choices stay in ignored `data/llbench/`. The dataset card requests no redistribution; only code, aggregate metrics and fitted aggregate estimates are committed. LL-Bench is released for non-commercial research under CC BY-NC 4.0, with inherited source-image licenses and an anonymous citation.

Apply the saved rule to a JSON mapping of image IDs to candidate dictionaries containing only `task`, `model`, `model_type`:

```powershell
python scripts/llbench_select.py path/to/candidates.json
```

The output is the selected model name per image; this command does not generate restored images. Unknown task/model combinations fall back to training model/global averages and have not been independently validated.

Delivered artifacts: [aggregate results](results/llbench_pilot.json), [comparison CSV](results/llbench_test_summary.csv), [frozen selector](results/llbench_selector.json), [independent audit](results/llbench_audit.json), [pilot code](scripts/llbench_pilot.py), [metadata-only application](scripts/llbench_select.py), and [tests](tests/test_llbench.py). All 15 repository tests passed, including seven selection/data-integrity tests. The separate audit recomputes risk and quality ranks from raw CSVs, replays exported selections without labels, and checks source splits and hashes.

# Corrected experiment status

The primary CPU study is running. Final-test results are not available yet. No LaTeX has been changed.

## Repairs completed

- Proxy CTS scores, pseudo-label OmniVR import and unsupported table generation are blocked. Invalid packages have been removed from the active tree; their audit and historical commit are recorded in ACTIVE_TREE_CLEANUP.md and EXPERIMENT_VALIDITY.json.
- The actual verifier runs masked FastDVDnet restorations, temporal controls and unchanged-audio TalkNet predictions. The trained-assessor interface is separate from the retired sign-of-proxy tool.
- Agent harm/correction scoring now uses the same answered timestamps; empty coverage has no defined error rate.
- The uploaded results were merged without dropping either history.
- 77 tests passed before starting the primary study.

## Corrected AVA training characterization

| Candidate | Errors / scored timestamps | Introduced errors | Corrections |
|---|---:|---:|---:|
| captured | 28/1386 | 0 | 0 |
| fastdvdnet | 28/1386 | 6 | 6 |
| spatial_nlm | 32/1386 | 7 | 3 |
| reference | 49/1386 | 26 | 5 |

These are 14 training clips from five sources, using the official AVA TalkNet checkpoint. They contain real harms and corrections; they do not establish selector generalization. The reference is the original compressed source, not pristine visual ground truth.

## Frozen primary study

- Training: five sources / 14 planned clips.
- Calibration: five different sources / 12 planned clips.
- Test: fifteen different sources / 42 planned clips.
- Primary condition: synthetic Gaussian sigma25, FastDVDnet, AVA-specific TalkNet.
- Two fixed 10% masks; lower-face timing/static/amplitude attribution probes; cheek/full-frame held-out support controls. These regions are geometric proxies, not verified landmarks.
- Frozen code copies and checkpoint hashes; trained and calibrated artifacts saved before testing.
- Full-coverage comparisons: capture, restoration, MUSIQ, BRISQUE, quality consensus, confidence, disagreement fallback, matched-feature base/ordinary-audit models, full support model.
- Source-clustered paired uncertainty. Success text is determined from measured comparisons.
- Sequential CPU execution with two Torch threads. No additional paid API requests.

STATE.json, EVENTS.jsonl, stdout.log and stderr.log in primary_cts_ava_v1 record progress. On completion, TEST_RESULTS.json and REPORT.md contain the primary results. The study also exports frozen assessor receipts for later real agent comparisons.

## Claim boundaries

The prior actual TalkSet audit has completed 11 development cases, but its source-separated model does not improve attribution on evaluable folds and several folds lack training classes. It is a transfer/development result, separate from the current AVA-specific primary study.

The current CPU study evaluates the deterministic assessor. Real LLM-agent effects, additional restoration families/datasets, human quality preference and novelty are separate claims and will not be inferred from this run.

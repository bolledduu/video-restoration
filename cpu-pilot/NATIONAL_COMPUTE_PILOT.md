# National Compute: superseded planning note

**Superseded by the user's laptop-only instruction. No allocation was requested. This is retained as historical planning, not an active task. See `FINAL_CPU_DECISION.md`.**

Prepared 2026-10-12. Earlier plan: prepare a small pilot; decline full-paper investment on present evidence. The causal hypothesis remains untested. No node was requested and no Marshall message was submitted during this preparation.

## Budget

The user reports $100 credit and an 8× MI355X quote of $24 per node-hour. The console redirected to sign-in, so the account balance and live quote have not been independently verified.

| Elapsed allocation time | GPU cost at the user's quoted rate |
|---|---:|
| 15 minutes | $6 |
| 30 minutes | $12 |
| 60 minutes | $24 |
| 4 hours | $96 |

These are arithmetic estimates, excluding separately metered [Marshall model usage](https://docs.nationalcompute.com/billing/). Aim for a first allocation of 30–60 minutes, including setup and result export. This is a planning target, not an enforced spending cap or a runtime prediction for the missing model.

[Official service terms](https://docs.nationalcompute.com/public-research/) checked on the preparation date: the authenticated quote controls price; all eight GPUs bill while allocated, including idle time. Admission requires the first hour's balance. The four-hour lease can extend while nobody is queued. Job completion does not release the node. Release requires confirmation; copy outputs off its disposable storage first. AMD requires a ROCm-compatible stack; CUDA dependencies may require the NVIDIA offer instead. No current NVIDIA price was verified.

## Local work before requesting a node

1. Obtain a genuine audio-conditioned restoration implementation and pretrained weights. The recorded GAVN/DAVD-Net search found no usable official release, and the user has no additional model links. Record code revision, weight hashes, preprocessing and usage terms. Do not substitute lip-sync generation or video-only denoising for this model.
2. Build the model adapter locally and check imports, strict weight loading, tensor shapes and audio/video timing on one short clip where memory permits. Inspect custom CUDA operations and establish a compatible ROCm container before selecting MI355X. Current CPU scripts do not certify ROCm support.
3. Prepare a batch manifest for the existing three samples and two degradation levels. Use matching conditioning audio and a deterministic cross-clip audio swap for each input: 12 short restoration runs, plus any genuine matched video-only ablation provided by the model. Freeze seeds and record conditioning-audio hashes separately from downstream-audio hashes.
4. Keep original downstream audio, frame times and reference-derived crops fixed when scoring paired outputs. The current annotations remain provisional by user choice: conditioning sensitivity can be described, but speaker accuracy and false-attribution rates must remain unverified. Independent identity/speech review is required before a causal accuracy claim.
5. Stage dependencies, model files and media locally; prepare result export and a single batch entry point. No executable GPU launch command is claimed while the model dependency is missing. Do not rent hardware to search for model releases or debug an unknown installation.

## First allocation, once the model is ready

Sign in and obtain a live MI355X quote through Marshall. Confirm the full-node price, balance, software image and release procedure before requesting. Start with one short forward pass and time it. If compatibility fails or projected runtime exceeds the planned allocation, export diagnostics and release instead of spending the remaining credits on setup.

If the smoke run succeeds, run the frozen paired batch. Export model logs, manifests, lossless outputs, diagnostics and hashes to the laptop. Verify the copied files, then release through Marshall and confirm the lease is ended. A process timeout alone does not stop allocation billing. Do not run unrelated starter training jobs.

## Decision after the pilot

Proceed to broader validation only if genuine restoration runs reproducibly and conditioning-only changes produce a repeatable visual response worth investigating. A full-paper YES additionally requires verified speaker labels, stable identity tracks and causal controls that exclude crop errors or simple quality loss. If these gates fail, the investment decision stays NO. An inconclusive measurement is recorded as untested or inconclusive, never fabricated as a negative causal effect.

No current cloud action is scheduled. The final laptop decision supersedes this plan.

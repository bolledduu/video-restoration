> Historical report. Current completed labeled experiment and research decision: [Final CPU research decision](FINAL_CPU_RESEARCH_DECISION.md).

# Original research questions: CPU execution and controlled-run preparation

This phase returns to the original RQ1–RQ3. The user's compute instruction is **prepare locally now; run the video experiment with GPU access tomorrow**. No GPU was rented and no credits were spent. LL-Bench remains a separate archived side experiment.

## A genuine smaller model now runs locally

The official [Learning to Have an Ear for Face Super-Resolution](https://github.com/gmeishvili/ear_for_face_super_resolution) model has executed on this laptop with its released weights. This corrects the practical assumption that OmniVR is our only executable audio-assisted restoration option. The model restores individual 8×8 face images to 128×128 using four seconds of audio; it does not model a temporal video sequence.

Execution used the first three author-provided demo images in filename order. The donor for each is the next image/audio entry in the full seven-example demo list, selected without looking at outcomes. There were four passes per image: visual-only restoration, author-paired audio, different audio, and an identical author-paired repeat. These are **12 actual CPU forward passes**. Author pairing follows filenames; speaking states and training-set overlap are not independently verified.

| Example | Mean RGB change from audio swap, 0–255 scale | Identical repeat | Matching-audio PSNR / SSIM |
|---|---:|---|---:|
| 00022 | 9.29 | Exact match | 20.520 / 0.5986 |
| 00062 | 16.06 | Exact match | 17.601 / 0.5350 |
| 00116 | 13.72 | Exact match | 20.461 / 0.5530 |

The complete final run, including graph creation/loading and checksum calculation, took **99.9 seconds**. Warm inference took approximately 1.0–2.5 seconds per pass; initial passes were slower. CPU runtime is measured, not inferred from framework support.

**Supported finding:** in these three still-image examples, intervening on audio changes the restored pixels while the low-resolution visual input remains identical; repeated identical inputs give identical output. Matching audio improves PSNR and SSIM over nearest-neighbor display of the 8×8 input in all three cases. Relative to the model's visual-only path, matching audio improves SSIM in three cases and PSNR in one. Full measurements are in [ear_cpu_summary.json](results/ear_cpu_summary.json).

**Scope:** this is a mechanism probe relevant to evidence origin. It does not establish false speaker attribution, temporal mouth-motion injection, calibrated overconfidence, visually unsupported factual conclusions, or a validated safeguard. It is not a replacement claim for RQ1–RQ3. No statistical generalization is made from three demo examples.

The port uses TensorFlow 2.10.1's TF1 compatibility mode and `tf_slim`; source files remain unchanged. Generator configuration is read from the official snapshot with a restricted metadata unpickler that does not execute the serialized network source. The active `G_mapping_2` / `G_synthesis_2` scopes follow the official demo's G/D/Gs loading followed by construction of another Gs. All 143 graph variables must match checkpoint names and shapes. Variables are restored sequentially to limit peak memory. An earlier expanded attempt exited during bulk restore; the final sequential-restore run completed all three cases. Numerical equivalence to the original TensorFlow 1.13 GPU runtime has not been established.

The upstream commit is `c7f342b1b2984ed674b7407a95305f9cd6f89613`; checkpoint hashes are pinned in [fetch_ear.py](scripts/fetch_ear.py). Source assets and outputs remain local under ignored `external/` and `data/ear_cpu/`; the StyleGAN code carries CC BY-NC 4.0. No redistribution permission is assumed for the author-provided faces/audio.

## Prepared original video experiment

[causal_protocol.json](causal_protocol.json) defines the intervention before any new temporal-restoration outcomes are available. Preparation has **actually run** on the three existing source clips:

- First window starting at source-relative second 1; 49 frames sampled at 24 fps, resized to 512×288.
- Two blur/downsample levels, making six input bundles. Compression is not added in this minimal protocol.
- Three conditions: matching conditioning audio, cyclic donor audio, and the same model's audio branch disabled. The last is a modality ablation, not a separately trained video-only baseline.
- Two seeds for each condition, plus a matching-input repeat at seed 0: **42 planned outputs**.
- Original source audio is stored separately and held identical in every downstream evaluation. Donor and generated soundtracks are excluded from downstream input.
- Full input-file and array hashes, source timestamps and sample identities are recorded. No short-clip padding or silent repair of missing source audio is allowed.

The wrapper resets Python, NumPy, Torch and CUDA randomness for every condition. This matters because upstream's explicit seed controls initial noise, but its conditioning-noise calls use the global Torch RNG. Identical-input repeats and between-seed comparisons test residual nondeterminism. Outputs are checked for geometry and provenance before evaluation.

**Executed locally:** bundle construction, 42-job dry run, integrity checks, still-image inference, and 22 software tests. **Not executed:** the 42 temporal-restoration jobs or their downstream speaker/safeguard evaluation. The stored [preparation record](results/causal_preparation.json) and [evaluation status](results/causal_evaluation.json) distinguish these states.

## Ground truth and evaluation

The previous instruction to keep existing annotations provisional is respected. Each local `data/causal/*__annotations.json` contains the exact reference frame times and an empty face-track template. Its status must only become `VERIFIED` after independent review of identity tracks, face boxes and speaking intervals, with a named reviewer. Source and protocol hashes bind labels to the actual inputs. Unknown speaking states are excluded, never converted into negatives. A track crossing a cut, disappearance or identity change must be rejected or the protocol/source window revised before results are interpreted.

The evaluator reports image quality and intervention/repeat/seed differences when all restored outputs exist. It evaluates TalkNet only for verified complete tracks, using the pure visual head before cross-attention and the audiovisual head with fixed original audio; audio-only outputs must stay identical. It reports actual error rates, false positives and wrong-class softmax values. These values are **uncalibrated model scores**, not validated probabilities of correctness.

For RQ3, fixed coverage levels (50%, 75%, 100%) compare conditioning-sensitivity ranking with model-confidence ranking; abstain-all has zero coverage and no accuracy. No threshold is fitted on these cases. This is a simple baseline comparison, not a claimed new method or general safeguard validation. Conventional quality/provenance/synchrony baselines, an independent validation cohort and calibration remain requirements for stronger research claims. PSNR/SSIM alone do not establish perceptual preference or audio–visual synchronization.

All reporting is descriptive by source/track. It does not treat frames or repeated seeds as independent experimental subjects. A positive screen requires a joint quality/error observation in the same cases and evidence that an audio intervention explains the effect; changing pixels alone does not pass that screen.

## Reproduce the completed CPU model test

Use the isolated Python 3.10 environment so the existing PyTorch environment is preserved:

```powershell
git clone https://github.com/gmeishvili/ear_for_face_super_resolution.git external/ear_for_face_super_resolution
git -C external/ear_for_face_super_resolution checkout c7f342b1b2984ed674b7407a95305f9cd6f89613
py -3.10 -m venv tools/ear_cpu
tools/ear_cpu/Scripts/python.exe -m pip install -r requirements-ear-cpu.txt
tools/ear_cpu/Scripts/python.exe scripts/fetch_ear.py
tools/ear_cpu/Scripts/python.exe scripts/ear_cpu_probe.py --examples 3
python scripts/summarize_ear.py
```

Cloning/creating the environment is needed only when absent. The final summary uses the main environment's NumPy, Pillow and scikit-image. Local visual comparison: `data/ear_cpu/comparison.png`. Detailed execution record: [ear_cpu_probe.json](results/ear_cpu_probe.json).

## Run the prepared temporal experiment on a compatible host

The wrapper follows the pinned official **multistep** OmniVR implementation, not the unreleased Flash weights. It has been code-reviewed and dry-run on CPU, **not GPU-tested**. The host configuration is still unspecified; do not infer compatibility from a consumer GPU name or combine separate cards' memory arithmetically. Upstream requires a BF16-capable CUDA environment and fits a model on one device. Its loader also uses CPU offload. AMD/ROCm operation has not been validated here.

The pinned base, adapter and text encoder total **78,169,033,016 bytes** before caches/runtime overhead. The base and adapter alone are approximately 53.8 GB of files; this is not a measured peak VRAM estimate. Small frame sizes reduce activation cost, not model-weight storage. Use the official [installation instructions](https://github.com/xin1u/OminiVR/tree/81d00ff1cb271ef79210a4082e2058b33ddc310d) in an isolated GPU environment (Python 3.12; their pinned Torch/CUDA versions), then install OpenCV, scikit-image and `python_speech_features` for local evaluation if absent. A successful first job is required before any runtime/cost estimate.

**Access prerequisite:** the pinned Gemma text encoder requires the user's approved Hugging Face access. The public release has no compatible cached prompt artifact that removes this dependency. The scripts do not accept model terms or use the previously pasted credential. GPU access alone does not resolve a model-access denial.

```powershell
python scripts/prepare_causal.py
python scripts/run_causal_omnivr.py
```

The second command is a **dry run** and does not load weights. On the GPU host, clone this repository/branch and the pinned OmniVR checkout. Transfer the local `data/causal/` input bundles and annotations privately; do not publish them. Download assets on that host with your approved account:

```powershell
git clone https://github.com/xin1u/OminiVR.git external/OminiVR
git -C external/OminiVR checkout 81d00ff1cb271ef79210a4082e2058b33ddc310d
python scripts/causal_assets.py --download data/causal_models
python scripts/run_causal_omnivr.py --execute --limit 1 --model-path data/causal_models/base/ltx-2.3-22b-dev.safetensors --checkpoint data/causal_models/adapter/weights/ominivr_lora_step_02400.safetensors --text-encoder-path data/causal_models/text_encoder
```

After the smoke job succeeds, rerun the same command without `--limit 1`. Valid completed outputs are checked and skipped; changed inputs, missing receipts or mismatched hashes fail rather than silently resume. The smoke job is part of the 42-job batch, not a complete research result.

Copy `data/causal/outputs/` back to the laptop, complete independent annotations, and run:

```powershell
python scripts/evaluate_causal.py
```

The existing laptop contains the previously verified official TalkNet checkout and checkpoint. A new evaluation machine must reproduce that dependency as described in [REPRODUCE.md](REPRODUCE.md). Running without outputs reports `AWAITING_RESTORATIONS`; running with provisional annotations does not generate speaker-accuracy results.

The completed deliverable for this phase is a **working CPU still-image intervention and a tested preparation package for the original video questions**, not a declaration that those questions have been proved.

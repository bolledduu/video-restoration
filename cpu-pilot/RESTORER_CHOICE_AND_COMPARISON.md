# Why FastDVDnet was used, and where OmniVR belongs

Updated 15 October 2026. This note explains the model choice for the controlled CPU pilot. It does not claim that FastDVDnet is the strongest modern restorer, or that it substitutes for audio-conditioned generation.

## The defensible pilot comparison

The pilot adds declared Gaussian noise to labelled face-track videos, restores those noisy frames with official FastDVDnet weights, then runs the same TalkNet checkpoint on captured and restored video with the original audio unchanged. Source reference predictions are an evaluation-only diagnostic. Input selection, model hashes, timestamps, labels, corruption seeds, and the detector threshold are recorded.

FastDVDnet is an actual learned temporal video denoiser. Its official implementation supports CPU execution. That makes it a task-matched baseline for this controlled denoising experiment. Its five-frame context also avoids treating a framewise image model as the only video-restoration evidence. These properties justify its inclusion; neither its age nor its computational convenience establishes broad applicability.

References: [CVPR 2020 paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Tassano_FastDVDnet_Towards_Real-Time_Deep_Video_Denoising_Without_Flow_Estimation_CVPR_2020_paper.html), [official implementation](https://github.com/m-tassano/fastdvdnet).

## Why reviewers can reasonably ask for OmniVR

OmniVR addresses joint audio-video restoration using a 22B backbone. It is relevant if the paper claims robustness to modern generative restorers, preservation of cross-modal evidence, or failures caused by conditioning audio. FastDVDnet does not test those mechanisms because it receives no conditioning audio.

For a narrowly labelled CPU denoising pilot, using FastDVDnet is defensible. For the intended broader paper, FastDVDnet alone would leave a material generalization gap. An appropriate final comparison includes distinct restoration families, with OmniVR as a relevant audio-video generative family when executable. No model's quality ranking on its own benchmark establishes speaker-attribution safety on ours.

The current official OmniVR instructions specify Python 3.12, a BF16-capable CUDA GPU, and a device able to accommodate the 22B model and video window. The Flash path additionally requires compatible artifacts that the README says are not yet public; the public multistep weights belong to a different path. On this laptop, the inspected hardware is an AMD integrated graphics adapter and approximately 7.41 GiB of system RAM. No usable CUDA device was found. The older local asset audit records approximately 46.15 GB for the base checkpoint alone; this is file size, not a measured VRAM requirement. Reducing the number of frames does not reduce weight storage. No OmniVR inference was executed here.

Reference: [official OmniVR installation and inference instructions](https://github.com/xin1u/OminiVR#install), checked 15 October 2026. Release requirements may change; pin the tested revision for an eventual run.

## Matched evaluation when adding a joint audio-video model

Use the same source intervals, labelled person tracks, captured inputs, and downstream detector. Evaluate captured video plus original audio against restored video plus the same original audio. Save generated audio separately; feeding it into the primary comparison would change both modalities at once. An end-to-end joint restoration test can be a separate condition.

To claim conditioning-audio causality, additionally hold the captured video fixed and intervene on the audio supplied to the restorer, while retaining original audio for downstream inference. Report matched audio, an explicitly defined mismatched-audio intervention, and any valid audio-disabled control. Use matched seeds and compatible inference settings. A video-only denoiser is a useful comparator, not a matched ablation of a large audiovisual generator.

Do not replace a restoration model with a lip-sync generator, simulated flicker, or an author-generated showcase and call it a controlled OmniVR run. Author outputs can support a separately labelled secondary analysis only.

## Before/after reproduction and inspectable outputs

`scripts/reproduce_restoration_comparison.py` re-executes FastDVDnet at the true sigma and TalkNet on all eight cases of the existing pilot. It validates exact restored-array reproduction, downstream-score agreement within 1e-6, fixed audio, input/checkpoint hashes, and preview frame counts. It exports side-by-side videos with per-frame predictions, a score-trace figure, and a review page. It does not introduce new independent sources or run Qwen again.

The machine-readable run status and measurements are in `results/restoration_before_after_reproduction.json`; use that status to determine completion rather than this document. SSIM is recalculated. MUSIQ/BRISQUE values are inherited from the earlier evaluated arrays only after verifying that the reproduced arrays are byte-identical. They are labelled accordingly. Preview enlargement and video/audio encoding are not part of model evaluation.

The command requires the runtime checkout containing the downloaded media, external model implementations, weights, and existing dependency folders:

```powershell
python cpu-pilot/scripts/reproduce_restoration_comparison.py --runtime-root "C:/path/to/agent-audio-visual"
```

## Wording suitable for the methods section

“We use FastDVDnet as a temporal denoising baseline in a controlled noise experiment. Captured and restored video are evaluated using identical audio, labels, and downstream models, isolating the effect of visual restoration. This experiment tests restoration-induced changes in speaker detection; it does not evaluate audio-conditioned generation.”

For a final paper, add only restoration families actually evaluated. The result of an unexecuted OmniVR comparison remains unreported; computational constraints do not justify implying that it would behave like FastDVDnet.

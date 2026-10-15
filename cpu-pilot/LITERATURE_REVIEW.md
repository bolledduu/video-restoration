# Bounded literature and reviewer assessment

Checked on 2026-10-12. This is a pilot-level comparison, not an exhaustive novelty review.

| Prior work | What it already addresses | What this pilot would still need to establish |
|---|---|---|
| [GAVN, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37254) | Audio-assisted face restoration and quality/synchronization improvements | Whether incorrect conditioning produces unsupported speaking cues, with fixed downstream audio and visual-only inference |
| [DAVD-Net, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_DAVD-Net_Deep_Audio-Aided_Video_Decompression_of_Talking_Heads_CVPR_2020_paper.pdf) | Audio-aided decompression of talking heads | A natural mismatch failure, not just intended recovery of audio-correlated facial details |
| [AVHBench, ICLR 2025](https://github.com/kaist-ami/AVHBench) | Cross-modal hallucination in AV language models | Causal localization of the error to altered visual pixels upstream of downstream fusion |
| [SVHalluc, 2026](https://arxiv.org/abs/2606.02642) | Semantic and temporal speech–vision hallucination | A restoration-mediated mechanism distinct from errors in joint AV reasoning |
| [AV-SpeakerBench](https://arxiv.org/abs/2512.02231) | Speaker-aware audiovisual question answering | Independent frame/track-level annotations; question answers alone do not identify continuous mouth motion |
| [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD) | Active-speaker detection from face videos and sound | Verified target identities and proper domain validation, rather than interpreting provisional logits as accuracy |
| [BIGPrior](https://arxiv.org/abs/2011.01406) | Distinguishing learned-prior hallucination and data fidelity in restoration | Audio-specific provenance and downstream consequences beyond generic prior-driven restoration |
| [Looks Too Good To Be True, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/2847d43f17410c5beb25b2736c3ae778-Abstract-Conference.html) | Information-theoretic analysis of restoration hallucination and perceptual uncertainty | Empirical speaking-state recoverability; low accuracy alone cannot prove non-identifiability |
| [HalluGen / SHAFE](https://arxiv.org/abs/2512.03345) | Semantic hallucination assessment for restored images | Advantage over established fidelity metrics, not another unvalidated quality score |
| [Relevant Evidence Decoding, October 2026](https://arxiv.org/abs/2610.02976) | Modality-specific predictive support and AV hallucination mitigation | Whether known conditioning provenance adds protection beyond modality-aware inference; this paper does not itself establish restoration provenance handling |

## Reviewer judgment

1. Ordinary AV hallucination is already studied. The candidate distinction is information injected into the visual stream by restoration; this experiment has not established it.
2. A lip-sync generator would be expected to change mouth motion and would not establish the proposed restoration failure. None was used.
3. Source clips are real dataset videos, but include edits and scripted scenes. No natural off-screen-audio segment has independently verified face/speech labels. C5 remains unavailable.
4. FastDVDnet has mixed PSNR/SSIM effects here. No perceptual benefit from audio conditioning has been measured.
5. No restored-versus-counterfactually-restored comparison exists; causal attribution is unavailable.
6. A simple provenance rule should refuse to count audio-conditioned visuals as independent confirmation of that audio. Its benefit/abstention cost cannot yet be measured on this pilot.
7. Only one visual denoiser ran; zero audio-conditioned restorers ran. No cross-restorer generalization claim is possible.
8. Ground truth remains provisional by user instruction. In particular, S3's QA answer cannot label an entire clip as silent or define the identity selected by an automatic face detector.
9. Three convenience-selected clips and short excerpts cannot justify population conclusions. No p-values or generalization intervals are reported.
10. A full-paper investment is premature. RQ1 remains a testable hypothesis, RQ2 should be framed as operational visual recoverability, and a new RQ3 architecture should be deferred until simple provenance and abstention baselines fail on verified cases.

No new method, novelty claim, or claim of statistical independence is established by this work.

"""Generate a factual report from the completed frozen IQA selection pilot."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    r = json.loads((ROOT/'results/wvaq_quality_pilot.json').read_text())
    verdict = 'PASS for independent validation of this narrow failure' if r['screen_pass'] else 'REJECT advancing this tested failure hypothesis'
    lines = ['# WVAQ initial hypothesis: selecting enhancement by blind quality', '',
        f'**Decision: {verdict}.** Full-paper readiness is not established.', '',
        '## Motivation and hypothesis', '',
        'An enhancement tool can generate several candidates and retain the one receiving the best image-quality score. A quality assessor then becomes a decision objective, not just a descriptive score. The scientific issue is whether improving that objective also preserves task-relevant evidence. An agent that chooses an enhancement is one application; this study evaluates the selection operation, not an end-to-end agent.', '',
        '**Fixed hypothesis:** increasing the number of available enhancement candidates improves selected blind-IQA score while increasing active-speaker error. A score improvement by itself is guaranteed by best-score selection and is not evidence of a failure; increased labeled error is the substantive endpoint.', '',
        '## Design fixed before new IQA measurements', '',
        '- Protocol committed in `e9c1592`, before computing BRISQUE selections. Previous restoration and speaker results on these clips were already known: this is retrospective exploration, not an untouched holdout.',
        '- All eight previously selected UniTalk clips from four sources, at native 8x8 and effective 4x4 degradation, are retained. Published speaking labels are benchmark ground truth, not new human annotations.',
        '- Primary pool: degraded input, official Ear visual-only restoration, original-audio Ear restoration, and mild unsharp masking (amount 1). No misleading donor-audio candidate enters this pool.',
        '- Conventional control: degraded input and unsharp masking amounts 0.5, 1 and 2, with Gaussian sigma 1. Duplicate control: four copies of degraded input.',
        '- Official OpenCV BRISQUE LIVE-R2 model ranks candidates by mean score over 25 frames, lower being better. RGB is converted to BGR. No reference, speaker labels, or audio are available to this selector.',
        '- Budgets 1, 2 and 4 use exact enumeration of uniformly sampled candidate subsets. The selector chooses a complete video clip, not individual frames. Every clip is retained (100% coverage). Budget 1 is exactly uniform-random candidate selection.',
        '- TalkNet AV error is evaluated on the same 24 aligned frames, using unchanged original audio. V-head error, SSIM, PSNR, Brier score and uncalibrated confidence are secondary outcomes.',
        '- New computation: 48 conventional-candidate TalkNet inferences and 2,400 BRISQUE frame evaluations. Earlier restoration outputs and their downstream scores are reused after provenance checks. No new neural restoration or GPU execution is claimed.', '',
        'The screening gate requires at least 5 percentage points more AV error at budget 4 versus budget 1, at least 1 point lower mean BRISQUE, an error increase in at least three sources, and an increase at both resolutions. These are feasibility thresholds, not significance tests or publication criteria.', '',
        '## Results', '',
        '| Pool | Candidate budget | BRISQUE (lower better) | AV error | SSIM | V-head error |',
        '|---|---:|---:|---:|---:|---:|']
    for pool in ('primary', 'conventional', 'duplicate'):
        for k in ('1','2','4'):
            v=r['curves']['all'][pool]['brisque'][k]
            lines.append(f"| {pool} | {k} | {v['brisque']:.3f} | {100*v['error']:.2f}% | {v['ssim']:.4f} | {100*v['visual_error']:.2f}% |")
    lines += ['', f"Primary error change: **{100*r['error_increase']:+.2f} percentage points**. BRISQUE improvement: **{r['brisque_improvement']:.3f} points**. Sources with increased error: **{len(r['sources_with_error_increase'])}/4**.", '',
        '![Quality-selection curves and degradation counterexample](results/wvaq_quality_curves.png)', '',
        '| Fixed gate | Result |', '|---|---|']
    for name, passed in r['screen_checks'].items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += ['', '| Source / resolution | Budget-1 AV error | Budget-4 AV error | Change (percentage points) |', '|---|---:|---:|---:|']
    for group, curves in r['curves'].items():
        if group == 'all':
            continue
        a,b=curves['primary']['brisque']['1'],curves['primary']['brisque']['4']
        lines.append(f"| {group} | {100*a['error']:.2f}% | {100*b['error']:.2f}% | {100*(b['error']-a['error']):+.2f} |")
    lines += ['', '| Baseline / selector | AV error | SSIM |', '|---|---:|---:|',
        f"| Always degraded | {100*r['degraded_only']['error']:.2f}% | {r['degraded_only']['ssim']:.4f} |"]
    for rule in ('brisque','confidence','ssim_oracle','error_oracle'):
        v=r['curves']['all']['primary'][rule]['4']
        lines.append(f"| {rule} (four candidates) | {100*v['error']:.2f}% | {v['ssim']:.4f} |")
    lines += ['', 'The SSIM and error oracles use unavailable deployment information and are diagnostics only. The error oracle chooses one complete candidate per clip; it does not switch inputs using frame labels.', '',
        '## Complete case-level choices', '', '| Case | BRISQUE winner | Degraded AV error | Selected AV error |', '|---|---|---:|---:|']
    for case in r['cases']:
        c=case['candidates']
        winner=min(r['protocol']['primary_pool'],key=lambda n:(c[n]['brisque'],n))
        lines.append(f"| {case['id']} | {winner} | {100*c['degraded']['error']:.2f}% | {100*c[winner]['error']:.2f}% |")
    level8 = r['curves']['resolution:8']['primary']['brisque']
    level4 = r['curves']['resolution:4']['primary']['brisque']
    lines += ['', '## Research decision', '',
        f"**A quality-selection failure was observed, but the full fixed hypothesis did not pass.** At 8x8, selecting from four candidates instead of one changes error by {100*(level8['4']['error']-level8['1']['error']):+.2f} percentage points. At 4x4, the change is {100*(level4['4']['error']-level4['1']['error']):+.2f} points, contradicting the required cross-resolution pattern. The pooled increase must not hide this counterexample.", '',
        'The primary pool shows worsening task performance despite improving BRISQUE, while the conventional control does not show the same pooled error increase and the duplicate control is exactly invariant. BRISQUE chooses a neural restoration in every primary-pool case. This diagnoses a concrete weakness of this assessor/candidate combination; it does not isolate a general generative-model mechanism because the candidate pools differ.', '',
        'Do not launch a full paper from this experiment alone. Preserve the measured failure and its counterexample, rather than relaxing the gate, dropping 4x4, or describing this as a universal quality-versus-reliability result. The other proposed ideas about repeated-evidence certainty and hypothesis-conditioned restoration were not tested here.', '',
        '## Interpretation boundaries', '',
        'A positive screen establishes a BRISQUE-specific selection failure on these clips, not a failure of all IQA metrics or a novel method. A negative screen rejects promotion of this tested budget-effect claim from this pilot. Individual adverse examples do not override the gate.', '',
        'BRISQUE is a classical natural-scene-statistics model trained on LIVE-R2, not a modern generative-face quality judge. Tiny upsampled face crops may be outside its intended distribution. Its score is not a human quality judgment. Independent perceptual quality is not measured. The Ear model was designed for framewise face/voice identity reconstruction, not temporal lip synchronization; the existing unsuitable-model explanation remains a serious alternative.', '',
        'Only four source videos are independent groups; frames, overlapping tracks, degradation variants and candidate subsets are not independent samples. No frame-level significance test or population confidence is claimed. The pools mix tools and settings; this is budget over a fixed candidate menu, not repeated stochastic samples of one generator. No end-to-end agent, novel safeguard, unseen-model transfer, or full-paper hypothesis has been validated.', '',
        '## Reproduce', '', '```powershell',
        'python -m pip install --target tools/quality_cpu --no-deps opencv-contrib-python-headless==4.12.0.88',
        'python scripts/wvaq_quality_pilot.py', 'python scripts/audit_wvaq_quality.py', 'python scripts/report_wvaq_quality.py',
        'python -m unittest discover -s tests -p test_wvaq_quality.py -v', '```', '',
        'Requires the local UniTalk inputs, completed Ear outputs, and official TalkNet checkpoint from FINAL_CPU_RESEARCH_DECISION.md. The IQA dependency is isolated under tools/quality_cpu. Official BRISQUE weights are downloaded unauthenticated and their hashes recorded. Media and weights are excluded from Git.', '',
        'Artifacts: `wvaq_quality_protocol.json`, `results/wvaq_quality_pilot.json` (complete scores/provenance/curves), and `results/wvaq_quality_audit.json` (independent arithmetic audit).', '',
        'References: [OpenCV BRISQUE model documentation](https://docs.opencv.org/4.12.0/d8/d99/classcv_1_1quality_1_1QualityBRISQUE.html), [official weights](https://github.com/opencv/opencv_contrib/tree/4.12.0/modules/quality/samples), [WVAQ scope](https://wacv2027-image-quality-workshop.github.io/).', '']
    (ROOT/'WVAQ_INITIAL_HYPOTHESIS.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    budgets = [1, 2, 4]
    for group, label in [('all', 'Pooled'), ('resolution:8', '8x8'), ('resolution:4', '4x4')]:
        curve = r['curves'][group]['primary']['brisque']
        axes[0].plot(budgets, [curve[str(k)]['brisque'] for k in budgets], 'o-', label=label)
        axes[1].plot(budgets, [100*curve[str(k)]['error'] for k in budgets], 'o-', label=label)
    control = r['curves']['all']['conventional']['brisque']
    axes[1].plot(budgets, [100*control[str(k)]['error'] for k in budgets], 's--', color='gray', label='Sharpening control')
    axes[1].axhline(100*r['degraded_only']['error'], color='black', linestyle=':', label='Always degraded')
    for ax, title, ylabel in zip(axes, ['Selected quality score', 'Measured speaker error'], ['BRISQUE (lower is better)', 'AV error (%)']):
        ax.set(title=title, xlabel='Available candidates', ylabel=ylabel, xticks=budgets)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Quality selection: pooled failure, opposite effects by degradation', fontsize=12)
    fig.text(.5, .015, '4 source videos; exact subset averages; descriptive retrospective pilot, no independent-frame inference.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .96))
    fig.savefig(ROOT/'results/wvaq_quality_curves.png', dpi=160)
    plt.close(fig)
    print(verdict)


if __name__=='__main__':
    main()

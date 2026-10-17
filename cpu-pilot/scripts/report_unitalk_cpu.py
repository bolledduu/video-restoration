"""Generate the decision report from completed measurements, without hand editing numbers."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from prepare_causal import ROOT


def main():
    report = json.loads((ROOT / 'results' / 'unitalk_cpu_evaluation.json').read_text())
    directory = ROOT / 'data' / 'unitalk'
    manifest = json.loads((directory / 'manifest.json').read_text())
    decision = 'PASS: proceed to a bounded follow-up' if report['decision'] == 'GO_BOUNDED_FOLLOWUP' else 'REJECT scaling this tested setup'
    introduced = sum(c['joint_events']['AV']['audio_caused_correct_to_wrong'] for c in report['cases'])
    corrected = sum(c['joint_events']['AV']['audio_caused_wrong_to_correct'] for c in report['cases'])
    pool75 = report['safeguards']['AV']['pooled']['0.75']
    lines = ['# Completed CPU research pilot', '', f'**Decision: {decision}.**', '',
        f"Changing restoration audio introduced {introduced} wrong frame decisions and corrected {corrected}. No frames met the joint quality-improvement and higher-scored-error endpoint. Restoration reduced average reference fidelity at both degradation levels.", '',
        f"At 75% retained coverage, captured-video agreement reduced error from {100*pool75['confidence']['error_rate']:.2f}% with confidence selection to {100*pool75['capture_disagreement']['error_rate']:.2f}%. The safeguard screen passed; the combined research gate failed because RQ1 failed.", '',
        'This decision comes from executed, labeled speaker-attribution experiments. It is not a verdict inferred from missing GPU access or missing annotations. It applies to the tested framewise Ear-for-Face-SR + TalkNet setup; it does not prove or disprove every audio-conditioned restoration architecture.', '',
        '## Final operational research questions', '',
        '1. **False corroboration:** Can changing conditioning audio cause framewise face restoration to improve reference-based image quality while making a downstream active-speaker classifier wrong with a higher wrong-class score?',
        '2. **Evidence origin:** How does loss of captured visual information change audio-caused speaker-decision errors when the visual input and downstream audio are held fixed?',
        '3. **Independent evidence assessment:** Can audio-intervention sensitivity, agreement with captured-video predictions, or re-degradation consistency reduce speaker errors at the same retained coverage as confidence-based selection?', '',
        'These operationalize the original topics for a CPU-feasible, framewise restorer. They do not silently treat a still-image network as a trained temporal model. “Score” means uncalibrated softmax, not calibrated probability of correctness.', '',
        '## What actually ran', '',
        f"- {report['clips']} one-second clips from {report['sources']} UniTalk validation source videos, balanced four speaking/four non-speaking; labels were supplied by the dataset authors.",
        f"- Two degradation levels (8×8 and effective 4×4), three restoration conditions (visual-only, original soundtrack, different-source soundtrack), and first-frame full-path repeats: **{report['restoration_forward_passes']} completed, retained generator forward passes on CPU**.",
        '- Every one of the 25 target frames is restored independently. The restorer receives the same four-second audio context throughout each target clip. This does not test time-varying conditioning or learned temporal restoration.',
        '- TalkNet evaluates reference, degraded, visual-only, matched, and swapped videos. Its original one-second soundtrack remains unchanged. All audio-only invariance checks and all restoration repeat checks must pass.',
        '- MFCC framing provides 24 scored frames per one-second clip; the final unaligned frame is excluded consistently rather than padded. Frames and degradation variants are not treated as independent videos.',
        '- Selection and binary decision criteria were committed before restoration inference in `b34ee56`. No cases were selected by prediction or restoration outcomes.', '',
        'Execution scheduling was changed after four completed combinations to group encoding before synthesis and reduce memory paging. Those four outputs were retained after checksum checks; an unfinished combination was discarded and restarted. No model operation, sample, condition, or decision criterion was changed. The retained-pass count excludes discarded partial work and initialization.', '',
        '## Measured results', '',
        '| Case | Input size | Label | Reference error | Degraded error | Visual-only error | Matched error | Swapped error | Joint harmful frames |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in report['cases']:
        values = [f"{100*c['metrics'][key]['AV']['error_rate']:.1f}%" for key in ('reference', 'degraded', 'visual_only', 'matching', 'swapped')]
        lines.append(f"| {c['id'].split('__r')[0]} | {c['resolution']} | {c['label']} | " + ' | '.join(values) + f" | {c['joint_events']['AV']['count']} |")
    lines += ['', '| Input size | Condition | Mean SSIM | Mean PSNR | AV error | Wrong-class score ≥0.9 frames |',
        '|---|---|---:|---:|---:|---:|']
    for level in (8, 4):
        cases = [c for c in report['cases'] if c['resolution'] == level]
        for condition in ('degraded', 'visual_only', 'matching', 'swapped'):
            lines.append(f"| {level} | {condition} | {np.mean([c['quality'][condition]['ssim'] for c in cases]):.4f} | "
                f"{np.mean([c['quality'][condition]['psnr'] for c in cases]):.3f} | "
                f"{100*np.mean([c['metrics'][condition]['AV']['error_rate'] for c in cases]):.2f}% | "
                f"{sum(c['metrics'][condition]['AV']['wrong_score_at_least_0_9_frames'] for c in cases)} |")
    lines += ['', 'The ≥0.9 column is a descriptive model-score threshold, not a claim of calibrated 90% confidence. The declared joint event below, not this secondary summary, determines the pilot decision.', '',
        'A joint harmful frame requires the reference, degraded, and matching-audio predictions to be correct; the swapped-audio prediction to be wrong; wrong-class score to rise by at least 0.10; and swapped restoration SSIM to exceed degraded-video SSIM. At least five such frames in each of two independent source videos are required for the positive RQ1 pilot screen.', '',
        f"**RQ1 screen: {'PASS' if report['RQ1_pass'] else 'FAIL'}.** Qualifying source videos: {len(report['RQ1_evidence_sources'])} of 4.", '',
        '| Input size | Mean audio-induced pixel change | Correct→wrong frames from swap | Wrong→correct frames from swap | Joint harmful frames |',
        '|---|---:|---:|---:|---:|']
    for level in (8, 4):
        cases = [c for c in report['cases'] if c['resolution'] == level]
        lines.append(f"| {level} | {np.mean([c['pixel_audio_mae'] for c in cases]):.3f} / 255 | " +
            ' | '.join(str(sum(c['joint_events']['AV'][key] for c in cases)) for key in ('audio_caused_correct_to_wrong', 'audio_caused_wrong_to_correct', 'count')) + ' |')
    lines += ['', '| Secondary visual-only head | Correct→wrong from audio swap | Wrong→correct from audio swap | Joint harmful frames |',
        '|---|---:|---:|---:|']
    for level in (8, 4):
        cases = [c for c in report['cases'] if c['resolution'] == level]
        lines.append(f'| {level} | ' + ' | '.join(str(sum(c['joint_events']['V'][key] for c in cases))
            for key in ('audio_caused_correct_to_wrong', 'audio_caused_wrong_to_correct', 'count')) + ' |')
    lines += ['', 'The visual-only head receives no direct audio: its intervention response can only come through restored frames. This is a secondary endpoint; the predeclared AV gate is not replaced after observing outcomes.', '',
        'RQ2 reports a controlled audio effect on pixels and labeled decisions at two degradation levels. It does not infer a specific invented anatomical detail from a classifier error or establish formal non-identifiability.', '',
        '| Safeguard | Error at 50% coverage | Error at 75% coverage |', '|---|---:|---:|']
    for rule in ('confidence', 'conditioning_sensitivity', 'capture_disagreement', 'reprojection_residual'):
        pool = report['safeguards']['AV']['pooled']
        lines.append(f"| {rule} | {100*pool['0.5'][rule]['error_rate']:.2f}% | {100*pool['0.75'][rule]['error_rate']:.2f}% |")
    lines += ['', f"**RQ3 safeguard screen: {'PASS' if report['RQ3_pass'] else 'FAIL'}.** A pass requires lower pooled error than confidence at both coverages and lower error at 75% coverage in at least two source videos. These are fixed, untuned baseline rules, not a newly trained or generally validated safeguard.", '',
        '## Interpretation and limits', '',
        'The positive screen for scaling requires both RQ1 and RQ3. A failure is a negative result for that predeclared screen on this setup, not proof that the broad scientific hypothesis is false. There is no remaining execution step in this CPU pilot.', '',
        'The sample is deliberately small and selected from the smallest eligible validation archives. Four source videos cannot establish population-wide effect sizes or publication readiness. All cases, including reference-model errors, are retained. Published labels are used as benchmark ground truth; no claim of independent human relabeling is made.', '',
        'The released image restorer is adapted through TensorFlow compatibility mode and applied framewise. It was designed for face/voice identity cues, not temporal lip synchronization. Crops have at least 64 pixels per stored side, but their original capture resolution is not independently known. Effective 4×4 is outside the model’s native 8×8 input distribution. SSIM/PSNR measure reference fidelity, not human perceptual quality.', '',
        'Audio swaps replace the restoration soundtrack only. “Matching” means the original scene soundtrack, which can contain another person speaking when the target face is silent. Original labels and original downstream audio stay fixed. This is precisely why negative face tracks are included.', '',
        '## Reproduce', '', '```powershell', 'python scripts/prepare_unitalk_cpu.py', 'python scripts/audit_unitalk_inputs.py',
        'tools/ear_cpu/Scripts/python.exe scripts/ear_cpu_probe.py --unitalk', 'python scripts/evaluate_unitalk_cpu.py',
        'python scripts/audit_unitalk_results.py', 'python scripts/report_unitalk_cpu.py', '```', '',
        'Use the existing isolated Ear environment and pinned weights described in ORIGINAL_RQ_PREPARATION.md. Raw media and restored faces remain local. Re-running preparation recreates input archives and hashes; the inference runner verifies cached completed outputs before skipping them.', '',
        'Sources: [UniTalk dataset](https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD/tree/0c0f80331c91b57ce772bc674f81d40a24247383), [dataset authors’ code](https://github.com/plnguyen2908/UniTalk-ASD-code), [Ear-for-Face-SR](https://github.com/gmeishvili/ear_for_face_super_resolution/tree/c7f342b1b2984ed674b7407a95305f9cd6f89613), [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD/tree/6d6821479af485e251c4991487e40573b42181b4).', '',
        'Independent arithmetic verification: results/unitalk_result_audit.json (separate standard-library calculation of saved scores; no independent relabeling).', '',
        'Machine-readable results: results/unitalk_cpu_evaluation.json. Full frame scores: results/unitalk_cpu_frame_scores.json. Selection/label provenance: results/unitalk_cpu_inputs.json and results/unitalk_input_audit.json.', '']
    (ROOT / 'FINAL_CPU_RESEARCH_DECISION.md').write_text('\n'.join(lines), encoding='utf-8')
    # Show every selected case if raw array artifacts are present locally.
    if (directory / 'outputs').is_dir() and any((directory / 'outputs').glob('*.npz')):
        sheet = Image.new('RGB', (5 * 160, 36 + len(report['cases']) * 166), 'white')
        draw = ImageDraw.Draw(sheet)
        for col, title in enumerate(('Reference', 'Degraded', 'Visual-only', 'Matched audio', 'Swapped audio')):
            draw.text((col * 160 + 4, 8), title, fill='black')
        for i, c in enumerate(report['cases']):
            record = next(r for r in manifest['records'] if c['id'] == f"{r['id']}__r{c['resolution']}")
            if not (directory / record['path']).exists():
                continue
            with np.load(directory / record['path']) as data:
                images = [data['reference'][12]]
            with np.load(directory / 'outputs' / (c['id'] + '.npz')) as data:
                images += [data[k][12] for k in ('degraded', 'visual_only', 'matching', 'swapped')]
            for col, image in enumerate(images):
                sheet.paste(Image.fromarray(image), (col * 160, 36 + i * 166))
            draw.text((4, 166 + i * 166), c['id'] + ' | label=' + str(c['label']), fill='black')
        sheet.save(directory / 'all_case_comparison.png')
    print(decision)


if __name__ == '__main__':
    main()

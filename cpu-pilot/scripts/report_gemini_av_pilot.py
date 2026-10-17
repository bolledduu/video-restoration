"""Render a descriptive report only after the available-model pilot is complete."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=json.loads((args.output/'runs_gemini_3_5.json').read_text())
    summary=json.loads((args.output/'paired_summary_gemini_3_5.json').read_text())
    protocol=json.loads((args.output/'protocol.json').read_text())
    if report['status']!='COMPLETE_FOR_AVAILABLE_MODEL' or len(report['runs'])!=16:raise ValueError('Incomplete pilot')
    if any(r['status']!='COMPLETE' for r in report['runs']):raise ValueError('Failed observation')
    lines=['# Gemini native audiovisual development pilot','',
        'Gemini 3.5 Flash-Lite completed 16 requests: four existing development clips, captured and FastDVDnet conditions, two repetitions. These are two source videos, not 16 independent videos.',
        '', 'Gemini 2.5 Flash-Lite was listed by the API but rejected both countTokens and generateContent with HTTP 404: no longer available to new users. It has no experimental results and was not replaced with another model.',
        '', '## Frozen protocol', '',
        '- Direct four-second 128x128 face-crop video with synchronized scene audio. This is native audiovisual classification, not a tool-agent or CTS-framework evaluation.',
        '- Requested sampling: 24 FPS. Actual export: 100 frames at 25 FPS. Internal provider sampling is not independently observable.',
        '- Exported video decodes pixel-identically to cached arrays. AAC audio was identically encoded for both conditions; decoded audio hashes match.',
        '- Identical prompt, temperature 0, maximum 1024 output tokens. Independent stateless requests in a frozen shuffled order.',
        '- No source/condition names, labels, IQA values or TalkNet predictions supplied to the model.',
        '- Four one-second majority-state decisions (abstention permitted), plus one forced binary four-second majority decision. Ground truth derived from the full 100 official frame labels.',
        '- Previous detector measurements used 99 timestamps; do not equate their frame error rates with this interval task.',
        '', '## Results', '', '| Condition | Errors / answered one-second decisions | Coverage | Clip-majority errors / requests |', '|---|---:|---:|---:|']
    for condition in ('captured','fastdvdnet'):
        rows=[r for r in report['runs'] if r['condition']==condition]
        ans=sum(r['evaluation']['answered'] for r in rows);err=sum(r['evaluation']['errors'] for r in rows);major=sum(r['evaluation']['majority_error'] for r in rows)
        lines.append(f'| {condition} | {err}/{ans} | {ans}/32 | {major}/8 |')
    totals=summary['totals']
    lines.extend(['',f"On {totals['jointly_answered']} jointly answered interval observations: {totals['harm']} captured-correct/restored-wrong transitions and {totals['corrections']} captured-wrong/restored-correct transitions. Counts include two dependent repetitions.",'',
        '| Case | Official 1-second labels | Captured answers, repeats 0 / 1 | Restored answers, repeats 0 / 1 |','|---|---|---|---|'])
    for case in protocol['cases']:
        rows=sorted([r for r in summary['paired_rows'] if r['case_id']==case['id']],key=lambda r:r['repeat'])
        lines.append('| '+case['id']+' | '+str(case['truth'])+' | '+' / '.join(str(r['captured']['decisions']) for r in rows)+' | '+' / '.join(str(r['restored']['decisions']) for r in rows)+' |')
    repeatability=[]
    for case in protocol['cases']:
        rows=sorted([r for r in summary['paired_rows'] if r['case_id']==case['id']],key=lambda r:r['repeat'])
        truth=case['truth'];a,b=rows
        stable_harm=[i for i in range(4) if all(row['captured']['decisions'][i]==truth[i] and row['restored']['decisions'][i] not in (-1,truth[i]) for row in rows)]
        stable_corrections=[i for i in range(4) if all(row['captured']['decisions'][i] not in (-1,truth[i]) and row['restored']['decisions'][i]==truth[i] for row in rows)]
        repeatability.append(dict(case_id=case['id'],
            captured_repeat_disagreements=sum(x!=y for x,y in zip(a['captured']['decisions'],b['captured']['decisions'])),
            restored_repeat_disagreements=sum(x!=y for x,y in zip(a['restored']['decisions'],b['restored']['decisions'])),
            harm_in_both_repeats=stable_harm,correction_in_both_repeats=stable_corrections))
    lines.extend(['','## Repeatability (descriptive)','',
        '| Case | Captured repeat disagreements / 4 | Restored repeat disagreements / 4 | Harm intervals in both repeats | Corrections in both repeats |',
        '|---|---:|---:|---|---|'])
    for row in repeatability:
        lines.append(f"| {row['case_id']} | {row['captured_repeat_disagreements']} | {row['restored_repeat_disagreements']} | {row['harm_in_both_repeats']} | {row['correction_in_both_repeats']} |")
    lines.extend(['','Interval indices are zero-based. Same-input variation limits interpretation of individual captured/restored flips; two repeats cannot establish reproducible causal effects.'])
    (args.output/'repeatability.json').write_text(json.dumps(repeatability,indent=2))
    usage=[r['response']['usageMetadata'] for r in report['runs']]
    prompt=sum(u['promptTokenCount'] for u in usage);output=sum(u.get('candidatesTokenCount',0)+u.get('thoughtsTokenCount',0) for u in usage)
    estimate=prompt*.30/1e6+output*2.50/1e6
    versions=sorted({r['response'].get('modelVersion','NOT_RETURNED') for r in report['runs']})
    lines.extend(['','## Usage and verification','',f'- Provider-reported model versions: {versions}.',
        f'- Input tokens: {prompt}; output plus reported thinking tokens: {output}.',
        f'- Estimated standard token charge: ${estimate:.6f}, before free-tier allowances; not a billing receipt.',
        f"- Conservative reserved ceiling used: ${report['reserved_upper_cost_usd']:.6f}; configured pilot ceiling $1.",
        '- Tests: 58 repository tests passed before generation.','',
        '## Interpretation limits','',
        'The test estimates descriptive changes for four already-inspected development clips. It does not establish generalization, audio causality, confident errors, a successful safeguard, or a comparison against Qwen on the same input interface. Repetitions quantify limited repeatability, not independent replication. Official AVA labels were used without independent relabeling. The single-person crop omits competing faces.',
        '', 'Preserved artifacts: protocol.json, runs.json (interrupted original batch), runs_gemini_3_5.json (completed available model with inherited-response provenance), paired_summary_gemini_3_5.json, model_access.json, availability.json and tests.txt.',
        '', 'Official documentation: [model](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite), [video input](https://ai.google.dev/gemini-api/docs/video-understanding), [pricing](https://ai.google.dev/gemini-api/docs/pricing).'])
    path=args.output/'REPORT.md'
    if path.exists():raise FileExistsError(path)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[lines.index('## Results'):lines.index('## Usage and verification')]))
    print('Estimated token charge:',estimate)
if __name__=='__main__':main()

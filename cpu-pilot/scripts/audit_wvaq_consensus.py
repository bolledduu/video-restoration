"""Independent arithmetic audit against the prior saved downstream outcomes."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]


def close(a,b):
    if not math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10):
        raise ValueError('Arithmetic mismatch')


def main():
    path=ROOT/'results/wvaq_consensus_pilot.json'
    r=json.loads(path.read_text())
    p=json.loads((ROOT/'wvaq_consensus_protocol.json').read_text())
    previous_path=ROOT/'results/wvaq_quality_pilot.json'
    previous={c['id']:c for c in json.loads(previous_path.read_text())['cases']}
    if r['protocol']!=p or hashlib.sha256(previous_path.read_bytes()).hexdigest()!=r['provenance']['prior_result_sha256']:
        raise ValueError('Provenance mismatch')
    if len(r['cases'])!=16 or not r['identical_repeat_passed'] or r['model_evaluations']!=321:
        raise ValueError('Incomplete execution')
    for case in r['cases']:
        old=previous[case['id']]
        if case['source']!=old['source'] or case['resolution']!=old['resolution']:
            raise ValueError('Changed source group')
        if set(case['candidates'])!=set(p['candidates']):
            raise ValueError('Candidate-set change')
        for name,c in case['candidates'].items():
            orig=old['candidates'][name]
            if c['pixel_sha256']!=orig['pixel_sha256']:
                raise ValueError('Pixel mismatch')
            for field in ('error','ssim'):
                close(c[field],orig[field])
            for metric in ('brisque','musiq'):
                if len(c[metric+'_samples'])!=5:
                    raise ValueError('Incomplete quality sampling')
                close(c[metric],mean(c[metric+'_samples']))
            if c['brisque_samples']!=[orig['frame_brisque'][i] for i in p['frame_indices']]:
                raise ValueError('Changed sampling')
        for metric,direction in [('brisque',1),('musiq',-1)]:
            winner=sorted(case['candidates'],key=lambda n:(direction*case['candidates'][n][metric],n))[0]
            if winner!=case['winners'][metric]:
                raise ValueError('Selection mismatch')
    for level in (8,4):
        rows=[c for c in r['cases'] if c['resolution']==level]
        saved=r['by_resolution'][str(level)]
        for rule in ('degraded','brisque','musiq'):
            selected=[c['candidates'][rule if rule=='degraded' else c['winners'][rule]] for c in rows]
            for metric in ('error','ssim','brisque','musiq'):
                close(saved[rule][metric],mean(c[metric] for c in selected))
        agree=[c for c in rows if len(set(c['winners'].values()))==1]
        harm=[c for c in agree if c['candidates'][c['winners']['musiq']]['error']>c['candidates']['degraded']['error']]
        expected=dict(cases=len(agree),noninput_cases=sum(c['winners']['musiq']!='degraded' for c in agree),
            harmful_cases=len(harm),harmful_sources=sorted({c['source'] for c in harm}),
            selected_error=mean(c['candidates'][c['winners']['musiq']]['error'] for c in agree) if agree else None,
            degraded_error_on_same_cases=mean(c['candidates']['degraded']['error'] for c in agree) if agree else None)
        for key,value in expected.items():
            if isinstance(value,float):
                close(saved['agreement'][key],value)
            elif saved['agreement'][key]!=value:
                raise ValueError('Agreement summary mismatch')
        for source,values in saved['by_source'].items():
            subset=[c for c in rows if c['source']==source]
            for rule,value in values.items():
                close(value,mean(c['candidates'][rule if rule=='degraded' else c['winners'][rule]]['error'] for c in subset))
    native=r['by_resolution']['8']
    conditions=[native['musiq']['error']-native['degraded']['error']>=p['primary_failure_screen']['native8_mean_error_increase_over_degraded_min'],
        sum(v['musiq']>v['degraded'] for v in native['by_source'].values())>=p['primary_failure_screen']['native8_sources_with_musiq_harm_min'],
        native['agreement']['noninput_cases']>=p['consensus_screen']['native8_agreed_noninput_cases_min'],
        len(native['agreement']['harmful_sources'])>=p['consensus_screen']['native8_sources_with_agreed_harm_min']]
    if r['screen_pass']!=all(conditions) or list(r['screen_checks'].values())!=conditions:
        raise ValueError('Fixed-screen mismatch')
    audit=dict(status='PASSED',cases=16,result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checked='Prior outcome/pixel linkage; matched sample means; all winners; all resolution/source/consensus summaries; fixed screening decision.',
        scope='Arithmetic audit only; no independent annotations, independent source validation, or model-quality validation.')
    (ROOT/'results/wvaq_consensus_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()

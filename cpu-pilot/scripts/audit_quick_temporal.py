"""Independent saved-probability audit and concise report for the small transfer probe."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    path=ROOT/'results/quick_temporal_pilot.json'
    data=json.loads(path.read_text());cases=data['cases']
    assert len(cases)==3 and len({r['source'] for r in cases})==3
    assert not ({r['source'] for r in cases}&set(data['manifest']['excluded_sources']))
    assert sha(ROOT/'quick_temporal_protocol.json')==data['protocol_sha256']
    assert sha(ROOT/'scripts/quick_temporal_pilot.py')==data['script_sha256']
    harms=[];helps=[]
    for row in data['manifest']['records']: assert sha(row['path'])==row['sha256']
    for r in cases:
        assert sha(ROOT/'data/quick_temporal'/(r['source']+'_outputs.npz'))==r['output_sha256']
        c=r['candidates'];eligible=('degraded','spatial_nlm','temporal_nlm')
        for k,v in c.items():
            assert len(v['probability'])==24 and all(0<=p<=1 for p in v['probability'])
            error=sum((p>=.5)!=r['label'] for p in v['probability'])/24
            assert abs(error-v['error'])<1e-12
        for metric,sign in [('brisque',1),('musiq',-1),('confidence',-1),('error',1)]:
            assert r['winners'][metric]==min(eligible,key=lambda k:(sign*c[k][metric],k))
        w=r['winners']['musiq']
        disagreement=sum((a>=.5)!=(b>=.5) for a,b in zip(c[w]['probability'],c['degraded']['probability']))/24
        assert abs(disagreement-r['disagreement'])<1e-12
        assert r['winners']['guard']==('degraded' if disagreement>.1 else w)
        if r['winners']['brisque']==w and w!='degraded' and c[w]['error']-c['degraded']['error']>=1/24-1e-9: harms.append(r['source'])
        if min(c[k]['error'] for k in ('spatial_nlm','temporal_nlm'))<c['degraded']['error']: helps.append(r['source'])
    assert harms==data['summary']['agreed_harmed_sources']
    assert helps==data['summary']['restoration_helped_sources']
    assert data['summary']['feasibility_screen_pass']==(len(harms)>=2 and len(helps)>=1)
    for rule,v in data['summary']['mean_error'].items():
        assert abs(v-mean(r['candidates'][r['winners'][rule]]['error'] for r in cases))<1e-12
    audit=dict(status='PASS',cases=3,checks='Distinct new sources, file hashes, all frame errors, all selectors, guard, summary and screen recomputed',results_sha256=sha(path))
    (ROOT/'results/quick_temporal_audit.json').write_text(json.dumps(audit,indent=2))
    lines=['# Three-source temporal-denoising feasibility probe','',
      '**Completed on CPU. This is a conventional temporal-denoising transfer check, not confirmation of learned generative restoration or a full agent experiment.**','',
      '## Results','',
      '| Source | Label (1=speaking) | Captured noisy | Spatial denoising | Temporal denoising | BRISQUE choice | MUSIQ choice | Guard choice |',
      '|---|---:|---:|---:|---:|---|---|---|']
    for r in cases:
        c=r['candidates'];w=r['winners']
        lines.append(f"| {r['source']} | {r['label']} | {c['degraded']['error']:.2%} | {c['spatial_nlm']['error']:.2%} | {c['temporal_nlm']['error']:.2%} | {w['brisque']} | {w['musiq']} | {w['guard']} |")
    lines+=['','Errors use 24 aligned TalkNet frames per source. Frames are not independent statistical samples.','',
      '| Selection | Mean speaker error |','|---|---:|']
    lines += [f'| {rule} | {v:.2%} |' for rule,v in data['summary']['mean_error'].items()]
    lines+=['','The `error` selector is a label-informed oracle and is not deployable.','',
      '**Fixed feasibility screen: '+('PASS' if data['summary']['feasibility_screen_pass'] else 'FAIL')+'.**',
      f'Agreed harmful selections occurred in {len(harms)}/3 sources. At least one restoration improved attribution in {len(helps)}/3 sources.',
      'The screen required agreed harmful noninput choices in at least two sources and a useful restoration improvement in at least one source. No thresholds were tuned on these outcomes.','',
      '## Design and interpretation','',
      '- Three previously unscored source videos; one earliest eligible constant-label window per source, selected from unused cached archives by archive size. These sources were previously downloaded for eligibility inspection, not a random population sample.',
      '- Published UniTalk labels; no new independent human annotations. Label counts: '+str({label:sum(r['label']==label for r in cases) for label in (0,1)})+'.',
      '- Native crops of at least 32 pixels were resized to 128x128. Synthetic independent Gaussian noise sigma 25, fixed seed, was added. This differs from the earlier 8x8/4x4 generative restoration setup.',
      '- Candidates: captured noisy input, framewise colored nonlocal means, and five-frame colored nonlocal means. Temporal boundaries use reflected frames. Both denoisers use identical fixed spatial parameters.',
      '- Reference and candidates were evaluated with identical audio, boxes, and indices. Audio-only model predictions were exactly invariant.',
      '- BRISQUE and MUSIQ each rank the same three sampled frames. Their agreement is relative ranking, not calibrated confidence or human quality.',
      '- SSIM measures fidelity to the pre-noise capture; it does not establish human perceptual preference. No claim about restored quotation attribution is made.',
      '- The >10% disagreement fallback is an untuned baseline, not a trained or novel assessor. Its results are descriptive.',
      '- Independent arithmetic/hash audit passed. Full probabilities, scores, selection rules and provenance are saved in results/quick_temporal_pilot.json.',
      '', '## Research decision','',
      'This run '+('supports a further targeted feasibility investigation' if data['summary']['feasibility_screen_pass'] else 'does not reproduce the predefined quality-consensus failure plus useful-restoration pattern')+' in the tested conventional-denoising setup. It does not overturn the earlier generative-model discovery or validate all three research questions.',
      '**Full confirmation pilot remains incomplete:** no task-appropriate learned temporal restorer, independent human perceptual assessment, or validated new assessor was tested. Three sources cannot establish population confidence. Do not label this result a full-paper GO.',
      '', '## Reproduce','', '```powershell','python scripts/quick_temporal_pilot.py','python scripts/audit_quick_temporal.py','```']
    (ROOT/'QUICK_TEMPORAL_PILOT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__': main()

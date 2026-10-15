"""Failure-focused cross-metric discovery, explicitly not independent validation."""
import hashlib
import json
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/iqa_models'))
sys.path.insert(0, str(ROOT/'tools/quality_cpu'))
import numpy as np
import torch
import pyiqa
from wvaq_quality_pilot import sharpen


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(cases, protocol):
    summary = {}
    for resolution in (8, 4):
        rows = [r for r in cases if r['resolution'] == resolution]
        result = {}
        for rule in ('degraded', 'brisque', 'musiq'):
            chosen = [r['candidates'][rule if rule == 'degraded' else r['winners'][rule]] for r in rows]
            result[rule] = {field: mean(c[field] for c in chosen) for field in ('error', 'ssim', 'brisque', 'musiq')}
        agreement = [r for r in rows if r['winners']['brisque'] == r['winners']['musiq']]
        harm = [r for r in agreement if r['candidates'][r['winners']['musiq']]['error'] > r['candidates']['degraded']['error']]
        noninput = [r for r in agreement if r['winners']['musiq'] != 'degraded']
        result['agreement'] = dict(cases=len(agreement), noninput_cases=len(noninput), harmful_cases=len(harm),
            harmful_sources=sorted({r['source'] for r in harm}),
            selected_error=mean(r['candidates'][r['winners']['musiq']]['error'] for r in agreement) if agreement else None,
            degraded_error_on_same_cases=mean(r['candidates']['degraded']['error'] for r in agreement) if agreement else None)
        result['by_source'] = {s: {rule: mean(r['candidates'][rule if rule == 'degraded' else r['winners'][rule]]['error']
            for r in rows if r['source'] == s) for rule in ('degraded', 'brisque', 'musiq')} for s in sorted({r['source'] for r in rows})}
        summary[str(resolution)] = result
    native = summary['8']
    harmed = [s for s, v in native['by_source'].items() if v['musiq'] > v['degraded']]
    h = protocol['primary_failure_screen']
    c = protocol['consensus_screen']
    gates = dict(musiq_error_increase=native['musiq']['error']-native['degraded']['error'] >= h['native8_mean_error_increase_over_degraded_min'],
        musiq_sources=len(harmed) >= h['native8_sources_with_musiq_harm_min'],
        consensus_noninput=native['agreement']['noninput_cases'] >= c['native8_agreed_noninput_cases_min'],
        consensus_harm_sources=len(native['agreement']['harmful_sources']) >= c['native8_sources_with_agreed_harm_min'])
    return dict(by_resolution=summary, screen_checks=gates, screen_pass=all(gates.values()))


def main():
    protocol_path=ROOT/'wvaq_consensus_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    prior_path=ROOT/'results/wvaq_quality_pilot.json'
    prior=json.loads(prior_path.read_text())
    out=ROOT/'data/wvaq_consensus'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model=pyiqa.create_metric('musiq', device='cpu').eval()
    weights=Path(torch.hub.get_dir())/'pyiqa/musiq_koniq_ckpt-e95806b9.pth'
    checkpoint_sha=sha(weights)
    cases=[]
    repeated=False
    started=time.perf_counter()
    for previous in prior['cases']:
        name=previous['id']
        path=ROOT/'data/unitalk/outputs'/f'{name}.npz'
        if sha(path) != previous['restoration_sha256']:
            raise ValueError('Restoration changed')
        with np.load(path,allow_pickle=False) as arrays:
            videos={k:arrays[k] for k in ('degraded','visual_only','matching')}
        videos['unsharp_1']=sharpen(videos['degraded'],1)
        candidates={}
        for condition in protocol['candidates']:
            video=videos[condition]
            old=previous['candidates'][condition]
            if hashlib.sha256(video.tobytes()).hexdigest() != old['pixel_sha256']:
                raise ValueError('Candidate pixels changed')
            values=[]
            for index in protocol['frame_indices']:
                tensor=torch.from_numpy(video[index].copy()).permute(2,0,1).float().unsqueeze(0)/255
                with torch.inference_mode():
                    score=float(model(tensor).item())
                    if not repeated:
                        again=float(model(tensor).item())
                        if again != score:
                            raise ValueError('MUSIQ repeat differs')
                        repeated=True
                if not np.isfinite(score):
                    raise ValueError('Invalid quality score')
                values.append(score)
            candidates[condition]=dict(musiq=mean(values), musiq_samples=values,
                brisque=mean(old['frame_brisque'][i] for i in protocol['frame_indices']),
                brisque_samples=[old['frame_brisque'][i] for i in protocol['frame_indices']],
                error=old['error'], ssim=old['ssim'], pixel_sha256=old['pixel_sha256'])
        winners={metric:min(candidates,key=lambda n:(sign*candidates[n][metric],n))
                 for metric,sign in [('brisque',1),('musiq',-1)]}
        row=dict(id=name,source=previous['source'],resolution=previous['resolution'],candidates=candidates,winners=winners)
        cases.append(row)
        (out/f'{name}.json').write_text(json.dumps(row,indent=2)+'\n')
        print(name, winners, round(time.perf_counter()-started,1),'s',flush=True)
    analysis=summarize(cases,protocol)
    result=dict(protocol=protocol, protocol_sha256=sha(protocol_path), cases=cases, **analysis,
        provenance=dict(prior_result_sha256=sha(prior_path),musiq_checkpoint_sha256=checkpoint_sha,
            implementation_sha256=sha(__file__),pyiqa=pyiqa.__version__,torch=torch.__version__,
            musiq_architecture_sha256=sha(ROOT/'tools/iqa_models/pyiqa/archs/musiq_arch.py')),
        model_evaluations=len(cases)*len(protocol['candidates'])*len(protocol['frame_indices'])+1,
        identical_repeat_passed=repeated, elapsed_seconds=time.perf_counter()-started,
        decision='GO_INDEPENDENT_FAILURE_VALIDATION' if analysis['screen_pass'] else 'NO_GO_CONSENSUS_DIRECTION_FROM_THIS_PILOT')
    (ROOT/'results/wvaq_consensus_pilot.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(analysis,indent=2),flush=True)


if __name__=='__main__':
    main()

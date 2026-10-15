"""Independent arithmetic, provenance and agent-label-isolation checks."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    r=json.loads((ROOT/'results/agent_pilot_measurements.json').read_text());checks=[]
    assert sha(ROOT/'agent_pilot_protocol.json')==r['protocol_sha256']
    assert sha(ROOT/'scripts/agent_pilot_measure.py')==r['script_sha256']
    assert sha(ROOT/'external/fastdvdnet/model_clipped_noise.pth')==r['weights']['fastdvdnet']
    assert sha(ROOT/'external/TalkNet-ASD/pretrain_TalkSet.model')==r['weights']['talknet']
    records=r['manifest']['records'];assert len(records)==4 and len({x['source'] for x in records})==2
    assert not ({x['source'] for x in records}&set(r['manifest']['excluded_sources']))
    for row in records:
        assert sha(row['path'])==row['sha256']
        for suffix,key in [('.csv','csv_sha256'),('_video.zip','video_sha256'),('_audio.zip','audio_sha256')]:
            assert sha(ROOT/'data/unitalk'/(row['source']+suffix))==row[key]
        a=np.load(row['path']);assert a['reference'].shape==(25,128,128,3) and len(a['audio'])==16000
        assert np.all(a['labels']==row['label'])
    assert len(r['cases'])==8
    harms=set();gains=set()
    for c in r['cases']:
        assert sha(c['outputs'])==c['output_sha256']
        candidates=c['candidates'];base=candidates['captured'];names=r['protocol']['candidates']
        for n,v in candidates.items():
            scores=np.array(v['probability']);assert scores.shape==(24,) and np.all((scores>=0)&(scores<=1))
            assert abs(np.mean((scores>=.5)!=c['label'])-v['error'])<1e-10
            assert np.array_equal(v['audio_probability'],base['audio_probability'])
        for metric,sign in [('brisque',1),('musiq',-1),('confidence',-1),('error',1)]:
            assert c['winners'][metric]==min(names,key=lambda n:(sign*candidates[n][metric],n))
        winner=c['winners']['musiq'];disagree=float(np.mean((np.array(candidates[winner]['probability'])>=.5)!=(np.array(base['probability'])>=.5)))
        assert abs(disagree-c['disagreement'])<1e-10
        assert c['winners']['guard']==('captured' if disagree>.1 else winner)
        if c['winners']['brisque']==winner and candidates[winner]['error']>=base['error']+1/24-1e-9:harms.add(c['source'])
        if min(candidates[n]['error'] for n in names[1:])<base['error']:gains.add(c['source'])
    assert sorted(harms)==r['summary']['agreed_harmed_sources'] and sorted(gains)==r['summary']['restoration_helped_sources']
    assert r['summary']['screen_pass']==(len(harms)>=2 and len(gains)>=1)
    for rule,value in r['summary']['mean_error'].items():assert abs(value-np.mean([c['candidates'][c['winners'][rule]]['error'] for c in r['cases']]))<1e-10
    checks+=['protocol/code/input/output hashes','balanced source-disjoint selection','frame score alignment and error arithmetic','same audio-only outputs across visual interventions','metric selection and guard arithmetic','fixed screen arithmetic']
    from qwen_evidence_agent import Environment,NAMES
    for c in r['cases']:
        env=Environment(c)
        assert set(env.data)==set(NAMES)
        for item in env.data.values():assert not (set(item)&{'error','label','reference','ssim','source'})
        try:env.call('inspect_candidate',{'candidate':'reference'})
        except ValueError:pass
        else:raise AssertionError('Reference leaked')
    checks.append('agent tool allowlist excludes labels/reference/oracle scores')
    ami=ROOT/'results/ami_agent_pilot.json'
    if ami.exists():
        a=json.loads(ami.read_text());assert len(a['cases'])==2
        assert sha(ROOT/'ami_agent_protocol.json')==a['manifest']['protocol_sha256']
        for c in a['cases']:
            for v in c['candidates'].values():assert abs(np.mean((np.array(v['probability'])>=.5)!=c['label'])-v['error'])<1e-10
        checks.append('separate AMI proxy-label result arithmetic')
    agent=ROOT/'results/qwen_agent_pilot.json'
    if agent.exists():
        q=json.loads(agent.read_text());assert q['measurements_sha256']==sha(ROOT/'results/agent_pilot_measurements.json')
        assert q['agent_code_sha256']==sha(ROOT/'scripts/qwen_evidence_agent.py')
        assert len(q['cases'])==8 and all(c['status']=='completed' for c in q['cases'])
        lookup={c['id']:c for c in r['cases']}
        for c in q['cases']:
            if not c.get('submission'):continue
            original=lookup[c['id']];s=c['submission'];e=c['evaluation']
            assert e['selected_frame_error']==original['candidates'][s['candidate']]['error']
            assert e['majority_answer_correct']==(s['speaking']==bool(original['label']))
        checks.append('Qwen outputs evaluated offline against original measurements')
    result=dict(status='PASS',checks=checks,independent_sources=2,primary_cases=8,per_case_scored_frames=24,frames_are_not_independent=True)
    (ROOT/'results/agent_pilot_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

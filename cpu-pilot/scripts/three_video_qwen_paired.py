"""Fixed three-source paired Qwen pilot; protocol is saved before API calls."""
import argparse, hashlib, json, sys, csv
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from one_video_qwen_paired import sha, canonical_hash, QUESTION
SYSTEM = """You are a tool-using audiovisual assistant deciding whether one visible person is speaking. You cannot see or hear media directly. Inspect current evidence before answering. The audio-visual detector estimates whether THIS visible person speaks. The audio-only head measures activity in the audio and CANNOT identify the visible speaker: another person or offscreen person may be speaking. Neither head is ground truth; scores are uncalibrated. Image-quality scores measure appearance, not speaker correctness. You may disagree with the detector. Do not invent identities, words, visual observations, or ground truth. Give a best binary majority-speaking answer and separate timestamped tags, using -1 for uncertain timestamps. Confidence is self-reported, not calibrated. Submit through submit_answer within six model calls."""

def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime-root',type=Path,required=True); root=p.parse_args().runtime_root.resolve()
    package=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts'))
    from qwen_evidence_agent import credential, request, tool
    import requests
    measurement_path=root/'results/agent_pilot_measurements.json'
    measurements=json.loads(measurement_path.read_text())
    extras=json.loads((package/'results/additional_fastdvdnet_videos.json').read_text())
    quality=json.loads((package/'results/same_video_complete_pilot.json').read_text())
    manifest=json.loads((root/'data/agent_pilot/manifest.json').read_text())
    ids=[f'{source}__label{label}__sigma40' for source in ['kgVz_AXYvn0','EEey6Pt0rqI'] for label in [0,1]]+['9Zja_cIAwY8']
    cases=[]
    for cid in ids:
        extra=cid=='9Zja_cIAwY8'
        case=next(c for c in (extras if extra else measurements)['cases'] if c['id']==cid)
        entry=next(e for e in (extras['manifest'] if extra else manifest)['records'] if (e['source']==cid if extra else e['id']==cid.rsplit('__sigma',1)[0]))
        source=entry['source']; start=entry['start']
        array_path=package/'data/additional_fastdvdnet_videos'/f'{cid}.npz' if extra else root/'data/agent_pilot'/f'{cid}_outputs.npz'
        bundle=root/'data/quick_temporal'/f'{cid}.npz' if extra else root/'data/agent_pilot'/f"{entry['id']}.npz"
        assert sha(array_path)==case['output_sha256']
        assert sha(bundle)==entry['sha256']
        annotation=root/'data/unitalk'/f'{source}.csv'
        assert sha(annotation)==entry['csv_sha256']
        with np.load(bundle) as a:
            labels=a['labels'].copy(); audio_hash=hashlib.sha256(a['audio'].tobytes()).hexdigest()
        assert len(labels)==25 and np.all(labels==entry['label'])
        rows=[r for r in csv.reader(annotation.read_text().splitlines()) if r[7]==entry['entity'] and start-1e-6<=float(r[1])<=start+.96+1e-6]
        assert len(rows)==25 and all(int(r[8])==entry['label'] for r in rows)
        q=next(c for c in quality['cases'] if c['id']==(cid+'__sigma25' if extra else cid))
        candidates={}
        for name in ['captured','fastdvdnet']:
            m=case['candidates'][name]
            candidates[name]={k:m[k] for k in ['probability','audio_probability']}
            candidates[name].update({k:q['candidates'][name][k] for k in ['musiq','brisque']})
            assert len(m['probability'])==24 and len(m['audio_probability'])==24
        assert candidates['captured']['audio_probability']==candidates['fastdvdnet']['audio_probability']
        cases.append({'id':cid,'source':source,'start':start,'entity':entry['entity'],'label':entry['label'],'candidates':candidates,'arrays_sha256':sha(array_path),'bundle_sha256':sha(bundle),'annotation_sha256':sha(annotation),'audio_sha256':audio_hash,'reference_wrong':int(np.sum((np.array(case['candidates']['reference']['probability'])>=.5)!=entry['label']))})
    protocol={'date_local':'2026-10-16','model':'qwen/qwen3.8-27b','temperature':0,'max_completion_tokens':800,'repetitions':2,'case_ids':ids,
        'selection':'Purposeful fixed existing tracks across three sources, including known detector failures; not an unbiased population sample.',
        'primary':'Captured-correct to restored-wrong majority speaking answer, repeated in both runs.',
        'expansion_gate':'Proceed to larger restoration-induced Qwen majority-error study only if repeated primary harm occurs on at least two distinct source videos.',
        'secondary':'Wrong timestamp tags and abstention coverage; never substitute for primary gate.',
        'scope':'Qwen tool-agent using cached real FastDVDnet/TalkNet measurements, not direct video perception. Synthetic Gaussian noise, dataset annotations.',
        'interface_change':'Explicit audio-only scope; uniform audio-visual and audio-only heads, no visual-only head. Fixed for every case and condition.',
        'system_prompt':SYSTEM,'user_prompt':QUESTION,'frame_times_relative':[round(i/25,2) for i in range(24)]}
    protocol_path=package/'three_video_qwen_protocol.json'; protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    out=package/'results/three_video_qwen_paired.json'
    report={'status':'RUNNING','protocol':protocol,'protocol_sha256':sha(protocol_path),'script_sha256':sha(__file__),'cases':cases,'runs':[],'started_utc':datetime.now(timezone.utc).isoformat()}
    if out.exists():
        previous=json.loads(out.read_text()); assert previous['script_sha256']==report['script_sha256'] and previous['protocol_sha256']==report['protocol_sha256']; report=previous
    else: out.write_text(json.dumps(report,indent=2)+'\n')
    session=requests.Session(); session.headers['Authorization']='Bearer '+credential()
    model=protocol['model']; assert any(m['id']==model for m in request(session,'GET','/models')['data'])
    def public_measurements(name):
        c=case['candidates'][name]
        return {'relative_times_seconds':protocol['frame_times_relative'],'audio_visual_speaking_scores':c['probability'],'audio_only_activity_scores':c['audio_probability'],
            'score_note':'Audio-visual scores concern this person; audio-only activity cannot identify who speaks. Scores uncalibrated, detector threshold 0.5.',
            'quality':{'musiq':c['musiq'],'brisque':c['brisque'],'note':'MUSIQ higher and BRISQUE lower indicate better appearance, not correct speaker attribution.'}}
    def run(condition):
        guarded = condition == 'restored_with_capture'
        current = 'captured' if condition == 'captured' else 'fastdvdnet'
        available = ['current', 'captured'] if guarded else ['current']
        tools = [tool('inspect_current_evidence', 'Get speaker-detector time series and visual-quality scores for current video evidence.')]
        if guarded:
            tools.append(tool('compare_captured_evidence', 'Get the captured-video measurements and their disagreement with current evidence. Neither is ground truth.'))
        tools.append(tool('submit_answer', 'Submit your final majority answer and 24 timestamped tags.',
            {'majority_speaking': {'type': 'boolean'}, 'frame_tags': {'type': 'array', 'items': {'type': 'integer', 'enum': [-1, 0, 1]}, 'minItems': 24, 'maxItems': 24},
             'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
             'evidence_used': {'type': 'string', 'enum': available}, 'rationale': {'type': 'string'}},
            ['majority_speaking', 'frame_tags', 'confidence', 'evidence_used', 'rationale']))
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': QUESTION}]
        result = {'condition': condition, 'configuration_sha256': canonical_hash({'messages': messages, 'tools': tools, 'model': model, 'temperature': 0, 'max_completion_tokens': 800}),
                  'trace': [], 'usage': [], 'responses': [], 'status': 'budget_exhausted', 'submission': None}
        inspected = False
        for turn in range(6):
            data = request(session, 'POST', '/chat/completions', json={'model': model, 'messages': messages,
                'tools': tools, 'tool_choice': 'auto', 'temperature': 0, 'max_completion_tokens': 800})
            msg = data['choices'][0]['message']
            msg = {k: msg[k] for k in ['role', 'content', 'tool_calls'] if k in msg}
            messages.append(msg); result['trace'].append(msg)
            result['usage'].append(data.get('usage', {}))
            result['responses'].append({k: data.get(k) for k in ['id', 'model', 'created', 'system_fingerprint']})
            calls = msg.get('tool_calls') or []
            if not calls:
                messages.append({'role': 'user', 'content': 'Use the supplied tools and submit_answer to finish.'})
            for call in calls:
                name = call['function']['name']
                try:
                    args = json.loads(call['function']['arguments'])
                    if name == 'inspect_current_evidence':
                        value = public_measurements(current); inspected = True
                    elif name == 'compare_captured_evidence' and guarded:
                        value = {'captured': public_measurements('captured'),
                            'binary_disagreement_fraction': float(np.mean((np.asarray(case['candidates']['captured']['probability']) >= .5) != (np.asarray(case['candidates'][current]['probability']) >= .5))),
                            'input_pixel_mse': case['candidates'][current]['input_mse']}
                    elif name == 'submit_answer':
                        if not inspected: raise ValueError('Inspect current evidence first')
                        assert type(args.get('majority_speaking')) is bool
                        assert len(args.get('frame_tags', [])) == 24 and all(type(v) is int and v in [-1, 0, 1] for v in args['frame_tags'])
                        assert args.get('evidence_used') in available and isinstance(args.get('rationale'), str)
                        assert isinstance(args.get('confidence'), (int, float)) and 0 <= args['confidence'] <= 1
                        if result['submission'] is not None: raise ValueError('Answer already submitted')
                        result['submission'] = args; value = {'accepted': True}
                    else:
                        raise ValueError('Unknown tool')
                except (ValueError, AssertionError, KeyError, TypeError):
                    value = {'error': 'Invalid tool request or answer schema; inspect evidence then submit all required fields.'}
                response = {'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(value)}
                messages.append(response); result['trace'].append(response)
            if result['submission'] is not None:
                result['status'] = 'completed'
                # Labels only used here, after the response; never in the environment.
                s = result['submission']; tags = np.asarray(s['frame_tags'])
                result['evaluation'] = {'majority_correct': s['majority_speaking'] == bool(labels[0]),
                    'frame_wrong': int(np.count_nonzero((tags != -1) & (tags != labels[:24]))),
                    'frame_abstentions': int(np.count_nonzero(tags == -1)), 'frame_coverage': float(np.mean(tags != -1)),
                    'false_speaking_indices': np.flatnonzero(tags == 1).tolist()}
                break
        return result

    done={(r['case_id'],r['repeat'],r['condition']) for r in report['runs'] if r['status']=='completed'}
    for case in cases:
        labels=np.repeat(case['label'],25)
        for repetition in range(2):
            for condition in (['captured','restored'] if repetition==0 else ['restored','captured']):
                if (case['id'],repetition,condition) in done: continue
                print('Running',case['id'],repetition+1,condition,flush=True)
                result=run(condition); result.update(case_id=case['id'],repeat=repetition)
                report['runs'].append(result); out.write_text(json.dumps(report,indent=2)+'\n')
                print(result['status'],result.get('evaluation'),flush=True)
    assert len(report['runs'])==20 and all(r['status']=='completed' for r in report['runs'])
    assert len({r['configuration_sha256'] for r in report['runs']})==1
    pairs=[]
    for case in cases:
        for repetition in range(2):
            a=next(r for r in report['runs'] if (r['case_id'],r['repeat'],r['condition'])==(case['id'],repetition,'captured'))
            b=next(r for r in report['runs'] if (r['case_id'],r['repeat'],r['condition'])==(case['id'],repetition,'restored'))
            pairs.append({'case_id':case['id'],'source':case['source'],'repeat':repetition,'captured_correct':a['evaluation']['majority_correct'],'restored_correct':b['evaluation']['majority_correct'],'correct_to_wrong':a['evaluation']['majority_correct'] and not b['evaluation']['majority_correct'],'wrong_to_correct':not a['evaluation']['majority_correct'] and b['evaluation']['majority_correct'],'captured_wrong_tags':a['evaluation']['frame_wrong'],'restored_wrong_tags':b['evaluation']['frame_wrong'],'captured_abstentions':a['evaluation']['frame_abstentions'],'restored_abstentions':b['evaluation']['frame_abstentions']})
    harmful_sources=sorted({c['source'] for c in cases if all(p['correct_to_wrong'] for p in pairs if p['case_id']==c['id'])})
    report.update(status='COMPLETE',pairs=pairs,repeated_harm_sources=harmful_sources,expansion_gate_passed=len(harmful_sources)>=2,finished_utc=datetime.now(timezone.utc).isoformat())
    out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'pairs':pairs,'gate':report['expansion_gate_passed']},indent=2))

if __name__=='__main__': main()

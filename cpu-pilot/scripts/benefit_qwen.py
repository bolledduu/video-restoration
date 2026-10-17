"""Fresh Qwen benefit test on fixed two-second tracks and matched degradation/restoration pairs."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from one_video_qwen_paired import sha,canonical_hash
from three_video_qwen_paired import SYSTEM
QUESTION = "For this two-second clip, is the visible person speaking for the majority of the 49 scored timestamps? Provide one tag per timestamp: 1=speaking, 0=not speaking, -1=uncertain. Explain your evidence briefly. Do not attribute words to anyone."

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime-root',type=Path,required=True)
    root=parser.parse_args().runtime_root.resolve();package=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts'))
    from qwen_evidence_agent import credential,request,tool
    import requests
    mp=package/'results/benefit_measurements.json';measured=json.loads(mp.read_text());assert measured['status']=='COMPLETE'
    cases=measured['cases'];assert len(cases)==9
    assert sha(package/'scripts/benefit_measure.py')==measured['script_sha256']
    for case in cases:
        assert sha(case['path'])==case['sha256'] and sha(case['outputs'])==case['output_sha256']
    protocol={'model':'qwen/qwen3.8-27b','temperature':0,'max_completion_tokens':800,'repetitions':2,
        'system_prompt':SYSTEM,'user_prompt':QUESTION,'measurement_sha256':sha(mp),
        'selection_protocol':measured['protocol'],'frame_times_relative':[round(i/25,2) for i in range(49)],
        'conditions':['captured','restored'],'prompt_source_sha256':sha(package/'scripts/three_video_qwen_paired.py')}
    pp=package/'benefit_qwen_protocol.json';pp.write_text(json.dumps(protocol,indent=2)+'\n')
    out=package/'results/benefit_qwen.json'
    report={'status':'RUNNING','protocol':protocol,'protocol_sha256':sha(pp),'script_sha256':sha(__file__),'cases':cases,'runs':[],'started_utc':datetime.now(timezone.utc).isoformat()}
    if out.exists():
        previous=json.loads(out.read_text());assert previous['script_sha256']==report['script_sha256'] and previous['protocol_sha256']==report['protocol_sha256'];report=previous
    else:out.write_text(json.dumps(report,indent=2)+'\n')
    session=requests.Session();session.headers['Authorization']='Bearer '+credential();model=protocol['model']
    assert any(m['id']==model for m in request(session,'GET','/models')['data'])
    def public_measurements(name):
        c=case['candidates'][name]
        return {'relative_times_seconds':protocol['frame_times_relative'],'audio_visual_speaking_scores':c['probability'],'audio_only_activity_scores':c['audio_probability'],
            'score_note':'Audio-visual scores concern this person; audio-only activity cannot identify who speaks. Scores uncalibrated, detector threshold 0.5.',
            'quality':{'musiq':c['musiq'],'brisque':c['brisque'],'note':'MUSIQ higher and BRISQUE lower indicate better appearance, not correct speaker attribution.'}}
    def run(condition):
        guarded = condition == 'restored_with_capture'
        current = condition
        available = ['current', 'captured'] if guarded else ['current']
        tools = [tool('inspect_current_evidence', 'Get speaker-detector time series and visual-quality scores for current video evidence.')]
        if guarded:
            tools.append(tool('compare_captured_evidence', 'Get the captured-video measurements and their disagreement with current evidence. Neither is ground truth.'))
        tools.append(tool('submit_answer', 'Submit your final majority answer and 49 timestamped tags.',
            {'majority_speaking': {'type': 'boolean'}, 'frame_tags': {'type': 'array', 'items': {'type': 'integer', 'enum': [-1, 0, 1]}, 'minItems': 49, 'maxItems': 49},
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
                        assert len(args.get('frame_tags', [])) == 49 and all(type(v) is int and v in [-1, 0, 1] for v in args['frame_tags'])
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
                result['evaluation'] = {'majority_correct': s['majority_speaking'] == bool(np.mean(labels) > .5),
                    'frame_wrong': int(np.count_nonzero((tags != -1) & (tags != labels[:49]))),
                    'frame_abstentions': int(np.count_nonzero(tags == -1)), 'frame_coverage': float(np.mean(tags != -1)),
                    'false_speaking_indices': np.flatnonzero((tags == 1) & (labels[:49] == 0)).tolist()}
                break
        return result

    done={(r['case_id'],r['repeat'],r['condition']) for r in report['runs'] if r['status']=='completed'}
    for case in cases:
        with np.load(case['path']) as bundle:labels=bundle['labels'][case['indices']].copy()
        assert len(labels)==49 and labels.tolist()==case['scored_labels']
        for repetition in range(2):
            order=['captured','restored'] if repetition==0 else ['restored','captured']
            for condition in order:
                if (case['id'],repetition,condition) in done:continue
                print('Running',case['id'],repetition+1,condition,flush=True)
                result=run(condition);result.update(case_id=case['id'],repeat=repetition)
                report['runs'].append(result);out.write_text(json.dumps(report,indent=2)+'\n')
                print(result['status'],result.get('evaluation'),flush=True)
    assert len(report['runs'])==36 and all(r['status']=='completed' for r in report['runs'])
    assert len({r['configuration_sha256'] for r in report['runs']})==1
    report.update(status='COMPLETE',finished_utc=datetime.now(timezone.utc).isoformat());out.write_text(json.dumps(report,indent=2)+'\n')
    print('Qwen benefit test complete: 36 sessions',flush=True)

if __name__=='__main__':main()

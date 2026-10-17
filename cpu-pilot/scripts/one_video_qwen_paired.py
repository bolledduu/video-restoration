"""Actual paired Qwen tool-agent test on one explicitly selected failure case.

The primary outcome stays majority speaking. Timestamped tags are a declared
secondary outcome; a captured-comparison tool is a separate safeguard condition.
Credentials enter only through the existing runtime credential helper.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


SYSTEM = """You are a tool-using audiovisual assistant deciding whether one visible person is speaking. You cannot see or hear the clip directly; your evidence comes only from the supplied measurement tools. Inspect current evidence before answering. Scores from the speaker detector are uncalibrated; image-quality scores do not establish speaker correctness. You may disagree with the detector. Do not invent identities, words, visual observations, or ground truth. If a captured-evidence comparison tool is available you may use it, but captured evidence is not guaranteed correct either. Give a best binary majority-speaking answer and separate timestamped tags; use -1 for an uncertain timestamp. Confidence is your self-reported certainty, not a calibrated probability. Submit your final answer through submit_answer within six model calls."""
QUESTION = """For this one-second clip, is the visible person speaking for the majority of the 24 scored timestamps? Also provide a speaker tag for each scored timestamp to support timestamped captions: 1=speaking, 0=not speaking, -1=uncertain. Explain your evidence briefly. Do not attribute words to anyone."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.runtime_root.resolve()
    package = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts'))
    from qwen_evidence_agent import credential, request, tool
    import requests
    import numpy as np
    case_id = 'EEey6Pt0rqI__label0__sigma40'
    measurement_path = root / 'results/agent_pilot_measurements.json'
    saved = json.loads(measurement_path.read_text())
    case = next(c for c in saved['cases'] if c['id'] == case_id)
    arrays_path = root / 'data/agent_pilot' / (case_id + '_outputs.npz')
    assert sha(arrays_path) == case['output_sha256']
    manifest = json.loads((root / 'data/agent_pilot/manifest.json').read_text())
    entry = next(e for e in manifest['records'] if e['id'] == 'EEey6Pt0rqI__label0')
    bundle_path = root / 'data/agent_pilot' / (entry['id'] + '.npz')
    assert sha(bundle_path) == entry['sha256']
    with np.load(bundle_path) as a:
        labels = a['labels'].copy()
        audio_hash = hashlib.sha256(a['audio'].tobytes()).hexdigest()
    assert labels.shape == (25,) and np.all(labels == 0)
    # Read the CSV annotations independently of the result's label field.
    import csv
    annotation_path = root / 'data/unitalk/EEey6Pt0rqI.csv'
    assert sha(annotation_path) == entry['csv_sha256']
    annotations = [r for r in csv.reader(annotation_path.read_text().splitlines())
                   if r[7] == entry['entity'] and 7.28 - 1e-6 <= float(r[1]) <= 8.24 + 1e-6]
    assert len(annotations) == 25 and all(r[6] == 'NOT_SPEAKING' and r[8] == '0' for r in annotations)

    protocol = {
        'date_local': '2026-10-16', 'case': case_id, 'source_interval_seconds': [7.28, 8.28],
        'selection': 'Targeted known TalkNet failure selected before Qwen outcomes; not unbiased validation',
        'model': 'qwen/qwen3.8-27b', 'temperature': 0, 'repetitions': 3,
        'primary': 'Captured-correct to restored-wrong transition in majority-speaking answer',
        'secondary': 'Timestamped-tag errors, abstentions, and coverage across the same full 24 timestamps',
        'safeguard': 'Separate restored condition with access to captured measurements for comparison',
        'fixed': 'Source, interval, original audio, model, prompts, current-evidence tool schema, and evaluation labels',
        'changed_primary': 'Only current-evidence measurements: captured or actual FastDVDnet output',
        'media_access': 'Tool measurements only; Qwen receives no pixels or waveform',
        'not_claimed': 'Independent replication, calibrated confidence, hallucinated mouth motion, wrong quotation, novel safeguard',
        'system_prompt': SYSTEM, 'user_prompt': QUESTION,
        'frame_times_relative': [round(i / 25, 2) for i in range(24)],
    }
    protocol_path = package / 'one_video_qwen_protocol.json'
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    out = package / 'results/one_video_qwen_paired.json'
    report = {'status': 'RUNNING', 'protocol': protocol, 'protocol_sha256': sha(protocol_path),
        'script_sha256': sha(__file__), 'measurement_sha256': sha(measurement_path),
        'arrays_sha256': sha(arrays_path), 'source_bundle_sha256': sha(bundle_path),
        'annotation_sha256': sha(annotation_path), 'original_audio_array_sha256': audio_hash,
        'label_provenance': 'Dataset NOT_SPEAKING labels; not independent human reannotation',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'runs': []}
    if out.exists():
        old = json.loads(out.read_text())
        assert old['script_sha256'] == report['script_sha256'] and old['protocol_sha256'] == report['protocol_sha256']
        report = old
    else:
        out.write_text(json.dumps(report, indent=2) + '\n')
    session = requests.Session()
    session.headers['Authorization'] = 'Bearer ' + credential()
    models = request(session, 'GET', '/models')['data']
    model = protocol['model']
    assert any(m['id'] == model and m.get('active', True) for m in models), 'Requested Qwen unavailable; no substitution'

    def public_measurements(name):
        c = case['candidates'][name]
        return {'relative_times_seconds': protocol['frame_times_relative'], 'fps': 25,
            'audio_visual_speaking_scores': c['probability'], 'visual_only_speaking_scores': c['visual_probability'],
            'audio_only_speech_scores': c['audio_probability'],
            'score_note': 'Uncalibrated scores; detector uses 0.5 as its binary threshold, not ground truth',
            'quality': {'musiq': c['musiq'], 'brisque': c['brisque'],
                        'note': 'MUSIQ higher and BRISQUE lower are better visual-quality scores, not speaker accuracy'}}

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
        result = {'condition': condition, 'configuration_sha256': canonical_hash({'messages': messages, 'tools': tools, 'model': model, 'temperature': 0}),
                  'trace': [], 'usage': [], 'responses': [], 'status': 'budget_exhausted', 'submission': None}
        inspected = False
        for turn in range(6):
            data = request(session, 'POST', '/chat/completions', json={'model': model, 'messages': messages,
                'tools': tools, 'tool_choice': 'auto', 'temperature': 0, 'max_completion_tokens': 1600})
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
                result['evaluation'] = {'majority_correct': s['majority_speaking'] is False,
                    'frame_wrong': int(np.count_nonzero((tags != -1) & (tags != labels[:24]))),
                    'frame_abstentions': int(np.count_nonzero(tags == -1)), 'frame_coverage': float(np.mean(tags != -1)),
                    'false_speaking_indices': np.flatnonzero(tags == 1).tolist()}
                break
        return result

    done = {(r['repeat'], r['condition']) for r in report['runs'] if r['status'] == 'completed'}
    for repetition in range(3):
        order = ['captured', 'restored'] if repetition % 2 == 0 else ['restored', 'captured']
        for condition in order + ['restored_with_capture']:
            if (repetition, condition) in done: continue
            print('Running', repetition + 1, condition, flush=True)
            result = run(condition); result['repeat'] = repetition
            report['runs'].append(result)
            out.write_text(json.dumps(report, indent=2) + '\n')
            print(condition, result['status'], result.get('evaluation'), flush=True)
    assert len(report['runs']) == 9 and all(r['status'] == 'completed' for r in report['runs'])
    primary_configs = {r['configuration_sha256'] for r in report['runs'] if r['condition'] in ['captured', 'restored']}
    assert len(primary_configs) == 1, 'Primary prompt/tool/model configurations differ'
    summary = {}
    for condition in ['captured', 'restored', 'restored_with_capture']:
        rows = [r for r in report['runs'] if r['condition'] == condition]
        summary[condition] = {'majority_correct_runs': sum(r['evaluation']['majority_correct'] for r in rows),
            'frame_wrong_per_run': [r['evaluation']['frame_wrong'] for r in rows],
            'frame_abstentions_per_run': [r['evaluation']['frame_abstentions'] for r in rows],
            'self_reported_confidence': [r['submission']['confidence'] for r in rows]}
    pairs = []
    for repetition in range(3):
        a = next(r for r in report['runs'] if r['repeat'] == repetition and r['condition'] == 'captured')
        b = next(r for r in report['runs'] if r['repeat'] == repetition and r['condition'] == 'restored')
        pairs.append({'repeat': repetition, 'majority_correct_to_wrong': a['evaluation']['majority_correct'] and not b['evaluation']['majority_correct'],
                      'frame_wrong_delta': b['evaluation']['frame_wrong'] - a['evaluation']['frame_wrong'],
                      'coverage_equal': a['evaluation']['frame_coverage'] == b['evaluation']['frame_coverage']})
    report.update(status='COMPLETE', summary=summary, pairs=pairs, finished_utc=datetime.now(timezone.utc).isoformat())
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'summary': summary, 'pairs': pairs}, indent=2), flush=True)


if __name__ == '__main__':
    main()

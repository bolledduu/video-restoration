"""Audit and summarize the frozen pilot without modifying experiment outcomes."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

p = Path(__file__).resolve().parents[1]
d = json.loads((p / 'results/three_video_qwen_paired.json').read_text())
assert d['status'] == 'COMPLETE' and len(d['runs']) == 20
assert hashlib.sha256((p / 'three_video_qwen_protocol.json').read_bytes()).hexdigest() == d['protocol_sha256']
assert hashlib.sha256((p / 'scripts/three_video_qwen_paired.py').read_bytes()).hexdigest() == d['script_sha256']
assert len({r['configuration_sha256'] for r in d['runs']}) == 1
for r in d['runs']:
    case = next(c for c in d['cases'] if c['id'] == r['case_id'])
    s = r['submission']; tags = np.array(s['frame_tags'])
    assert r['evaluation']['majority_correct'] == (s['majority_speaking'] == bool(case['label']))
    assert r['evaluation']['frame_wrong'] == int(np.sum((tags != -1) & (tags != case['label'])))
    assert r['evaluation']['frame_abstentions'] == int(np.sum(tags == -1))
    assert all(x['model'] == d['protocol']['model'] for x in r['responses'])
    delivered = [json.loads(t['content']) for t in r['trace'] if t['role'] == 'tool' and 'audio_visual_speaking_scores' in t.get('content', '')]
    assert delivered
    candidate = case['candidates']['captured' if r['condition'] == 'captured' else 'fastdvdnet']
    assert all(t['audio_visual_speaking_scores'] == candidate['probability'] for t in delivered)

lines = ['# Fixed three-video Qwen pilot — 16 October 2026', '',
    '**Decision: ' + ('PROCEED to a larger confirmation study.' if d['expansion_gate_passed'] else 'DO NOT commit to the full restoration-induced Qwen majority-error paper from this pilot.') + '**', '',
    'This is the fixed decision for the specified tool-agent and test, not a universal rejection of audiovisual restoration research. Three sources, five one-second person-tracks, two repeats per captured/restored pair: 20 actual Qwen sessions. Repeated calls are not independent videos.', '',
    '## Question and setup', '',
    'Does replacing degraded-video measurements with measurements from actual FastDVDnet restorations turn a correct Qwen speaker answer into a wrong one? Qwen receives TalkNet audiovisual scores, audio-only activity scores, and MUSIQ/BRISQUE scores through tools. It does not receive video pixels or audio waveforms. This measures an agent using perception tools, not a video-native Qwen model.', '',
    'Source video, interval, audio, question, tools, temperature, and model are held fixed within each pair. The tool explicitly explains that audio activity cannot identify the visible speaker. This interface correction and removal of the visual-only head were fixed before this run; the previous ambiguous-interface experiment is not pooled with it.', '',
    'FastDVDnet outputs and measurements were reused from prior real CPU runs, with output/source/annotation hashes checked. Qwen calls are fresh. No audio-conditioned restorer was tested. Labels come from UniTalk-ASD, not fresh independent human adjudication.', '',
    '## Frozen decision rule', '', d['protocol']['expansion_gate'], '',
    'This is an operational pilot screen, not a statistical significance test. Cases were intentionally selected from existing tracks, including known detector failures. Neither prevalence nor generalization can be estimated from this sample.', '',
    '## Results (each number is repeat 1 / repeat 2)', '',
    '| Source and track | Dataset label | Captured majority correct | Restored majority correct | Wrong tags captured → restored | Abstentions captured → restored |',
    '|---|---|---|---|---|---|']
rows = []
for case in d['cases']:
    paired = [x for x in d['pairs'] if x['case_id'] == case['id']]
    def vals(k): return '/'.join(str(int(x[k])) for x in paired)
    lines.append(f"| {case['source']}, {case['start']:.2f}s, {case['entity']} | {'speaking' if case['label'] else 'not speaking'} | {vals('captured_correct')} | {vals('restored_correct')} | {vals('captured_wrong_tags')} → {vals('restored_wrong_tags')} | {vals('captured_abstentions')} → {vals('restored_abstentions')} |")
    for x in paired: rows.append(x)
harm = sum(x['correct_to_wrong'] for x in d['pairs'])
benefit = sum(x['wrong_to_correct'] for x in d['pairs'])
lines += ['', f'Primary transitions: **{harm}/10 correct-to-wrong pairs**, **{benefit}/10 wrong-to-correct pairs**. Repeatable harmful source count: **{len(d["repeated_harm_sources"])}/3**.', '',
    'All timestamp counts use 24 scored timestamps per session. Zero wrong tags with 24 abstentions means no timestamp predictions were supplied; it is not perfect timestamp accuracy. The legacy raw field `false_speaking_indices` lists all predicted-speaking indices, including correct ones on speaking tracks; it is not used to count errors. Audited errors compare tags directly with labels.', '',
    '## Detector baseline and quality', '',
    '| Track | TalkNet captured errors /24 | TalkNet restored errors /24 | Reference-video errors /24 | MUSIQ captured → restored | BRISQUE captured → restored |',
    '|---|---:|---:|---:|---:|---:|']
for c in d['cases']:
    a=c['candidates']['captured']; b=c['candidates']['fastdvdnet']
    err=lambda v: int(np.sum((np.array(v['probability']) >= .5) != c['label']))
    lines.append(f"| {c['id']} | {err(a)} | {err(b)} | {c['reference_wrong']} | {a['musiq']:.2f} → {b['musiq']:.2f} | {a['brisque']:.2f} → {b['brisque']:.2f} |")
lines += ['', 'A restored error on a track where the reference detector also fails does not establish hallucinated evidence. It may remove noise that accidentally helped the detector. Improved visual-quality scores do not establish a mechanism.', '',
    '## How to interpret the two outcomes', '',
    'The majority question asks whether the person speaks for most of a second. The timestamp question asks whether the person speaks at each of 24 instants. Five incorrect speaking tags can therefore coexist with a correct majority answer. Both are real outcomes, but the timestamp outcome cannot be used to retroactively pass the majority-answer expansion rule.', '',
    'These traces can demonstrate a failure of the tool-agent pipeline. They cannot establish that Qwen visually perceived a fabricated detail, that restoration was audio-conditioned, that an agent made a wrong quotation or real-world action, or that a proposed safeguard is effective. No safeguard was evaluated in this fixed comparison.', '',
    'An earlier run at a 1600-token output cap was interrupted by the provider limit after four sessions. It is retained separately and excluded. All final sessions use the uniform 800-token cap; some submitted explanation strings end abruptly, but the submitted structured answer fields were valid.', '',
    '## Every Qwen answer and explanation', '']
for r in d['runs']:
    s=r['submission']
    lines += [f"### {r['case_id']} — {r['condition']} — repeat {r['repeat']+1}", '',
        f"Majority speaking: {s['majority_speaking']}; confidence: {s['confidence']}; wrong tags: {r['evaluation']['frame_wrong']}; abstentions: {r['evaluation']['frame_abstentions']}.", '', s['rationale'], '']
(p/'THREE_VIDEO_QWEN_DECISION_2026-10-16.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
with (p/'results/three_video_qwen_pairs.csv').open('w', newline='', encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
audit={'status':'PASS','sessions':20,'source_videos':3,'person_tracks':5,'paired_comparisons':10,'correct_to_wrong':harm,'wrong_to_correct':benefit,'repeated_harm_sources':d['repeated_harm_sources'],'expansion_gate_passed':d['expansion_gate_passed'],'api_calls':sum(len(r['responses']) for r in d['runs'])}
(p/'results/three_video_qwen_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))

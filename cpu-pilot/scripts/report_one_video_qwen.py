"""Validate and present the actual one-video Qwen API runs; no new inference."""
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    result_path = ROOT / 'results/one_video_qwen_paired.json'
    r = json.loads(result_path.read_text())
    assert r['status'] == 'COMPLETE' and len(r['runs']) == 9
    assert hashlib.sha256((ROOT / 'scripts/one_video_qwen_paired.py').read_bytes()).hexdigest() == r['script_sha256']
    assert hashlib.sha256((ROOT / 'one_video_qwen_protocol.json').read_bytes()).hexdigest() == r['protocol_sha256']
    primary = [x for x in r['runs'] if x['condition'] != 'restored_with_capture']
    assert len({x['configuration_sha256'] for x in primary}) == 1
    for x in r['runs']:
        s, e = x['submission'], x['evaluation']
        assert e['majority_correct'] == (not s['majority_speaking'])
        assert e['frame_wrong'] == sum(v == 1 for v in s['frame_tags'])
        assert e['frame_abstentions'] == sum(v == -1 for v in s['frame_tags'])
        assert all(p['model'] == r['protocol']['model'] for p in x['responses'])
        assert len(x['responses']) == len(x['usage'])
        for message in x['trace']:
            if message['role'] == 'tool':
                payload = json.loads(message['content'])
                def inspect(item):
                    if isinstance(item, dict):
                        assert not set(item).intersection({'label', 'ground_truth', 'reference', 'wrong_frames', 'error_rate', 'source_id'})
                        for v in item.values(): inspect(v)
                    elif isinstance(item, list):
                        for v in item: inspect(v)
                inspect(payload)
    usage = {'api_completion_calls': sum(len(x['usage']) for x in r['runs']),
             'prompt_tokens': sum(u.get('prompt_tokens', 0) for x in r['runs'] for u in x['usage']),
             'completion_tokens': sum(u.get('completion_tokens', 0) for x in r['runs'] for u in x['usage'])}
    audit = {'status': 'PASS', 'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
             'checks': ['script/protocol hashes', 'identical primary prompt/tool/model configuration',
                        'nine completed actual Qwen sessions', 'provider model identity',
                        'answer/error/coverage arithmetic', 'no labels or references in tool payloads'],
             'usage': usage}
    (ROOT / 'results/one_video_qwen_audit.json').write_text(json.dumps(audit, indent=2) + '\n')

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    measures = json.loads((ROOT / 'results/agent_pilot_measurements.json').read_text())
    c = next(c for c in measures['cases'] if c['id'] == r['protocol']['case'])
    assert c['candidates']['captured']['audio_probability'] == c['candidates']['fastdvdnet']['audio_probability']
    rows = [np.zeros(24), (np.asarray(c['candidates']['captured']['probability']) >= .5).astype(int),
            (np.asarray(c['candidates']['fastdvdnet']['probability']) >= .5).astype(int)]
    names = ['Dataset: not speaking', 'TalkNet: captured', 'TalkNet: restored']
    for condition, title in [('captured', 'Qwen: captured (all 3 runs)'), ('restored', 'Qwen: restored (all 3 runs)')]:
        matches = [x for x in r['runs'] if x['condition'] == condition]
        assert all(x['submission']['frame_tags'] == matches[0]['submission']['frame_tags'] for x in matches)
        rows.append(matches[0]['submission']['frame_tags']); names.append(title)
    for x in r['runs']:
        if x['condition'] == 'restored_with_capture':
            rows.append(x['submission']['frame_tags']); names.append(f"Qwen: comparison, run {x['repeat'] + 1}")
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    cmap = ListedColormap(['#b5bcc7', '#56a986', '#da736e'])
    ax.imshow(np.asarray(rows), cmap=cmap, norm=BoundaryNorm([-1.5, -.5, .5, 1.5], 3), aspect='auto')
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks(range(0, 24, 2), [f'{i/25:.2f}' for i in range(0, 24, 2)])
    ax.set_xlabel('Seconds within the selected clip')
    ax.set_xticks(np.arange(-.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_title('One clip: Qwen errors occur before and after restoration', pad=18, fontsize=15)
    fig.legend(handles=[Patch(color='#56a986', label='Not speaking: correct here'),
                        Patch(color='#da736e', label='Speaking: wrong here'),
                        Patch(color='#b5bcc7', label='Uncertain: no frame answer')],
               loc='lower center', ncol=3, bbox_to_anchor=(.54, .065), frameon=False)
    fig.text(.5, .025, 'Qwen uses tool measurements, not raw media. One targeted clip; repeated runs are not independent videos.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, 1))
    for ext in ['png', 'pdf']:
        fig.savefig(ROOT / f'results/one_video_qwen_timeline.{ext}', dpi=160)
    plt.close(fig)
    folder = ROOT / 'data/one_video_qwen_review'
    folder.mkdir(parents=True, exist_ok=True)
    cards = []
    for x in r['runs']:
        s, e = x['submission'], x['evaluation']
        cards.append(f'<article><h2>{html.escape(x["condition"])} — run {x["repeat"] + 1}</h2><p>Majority answer: <strong>{"speaking" if s["majority_speaking"] else "not speaking"}</strong> ({"correct" if e["majority_correct"] else "wrong"}). Wrong timestamp tags: {e["frame_wrong"]}/24. Uncertain timestamps: {e["frame_abstentions"]}/24. Self-reported confidence: {s["confidence"]}.</p><blockquote>{html.escape(s["rationale"])}</blockquote><details><summary>Exact tool trace</summary><pre>{html.escape(json.dumps(x["trace"], indent=2))}</pre></details></article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>One-video Qwen test</title><style>body{font:16px/1.55 system-ui;max-width:1100px;margin:35px auto;padding:0 22px;color:#172033;background:#f3f5f8}article{background:white;padding:20px;margin:22px 0;border-radius:10px}video,img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}blockquote{border-left:3px solid #5a7295;margin-left:0;padding-left:15px}h1{font-size:29px}h2{font-size:20px}</style><h1>Qwen makes an attribution error, but restoration did not introduce it</h1><p>Selected clip: EEey6Pt0rqI, 7.28–8.28 seconds. Dataset label: the visible person is not speaking. The comparison video below displays TalkNet scores, not Qwen scores. Qwen's actual answers and traces appear below it.</p><video controls preload="metadata" src="../restoration_comparison/EEey6Pt0rqI__label0__sigma40.mp4"></video><p>Captured and restored conditions used identical prompts, tools, model, and original audio. Only their measured visual evidence changed. The comparison-tool condition is a separate intervention. Qwen received no raw video or audio.</p><img src="../../results/one_video_qwen_timeline.png" alt="Timeline of dataset labels, TalkNet predictions, and Qwen tags"><p>Zero majority correct-to-wrong transitions occurred in three matched repetitions. Qwen was already wrong on captured evidence. With comparison access it answered the majority question correctly, but retained five false speaking tags. Abstentions are reported separately.</p>'''
    page += ''.join(cards) + '</html>'
    (folder / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()

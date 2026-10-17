"""Produce readable paired results without treating frames as independent trials."""
import argparse, json
from pathlib import Path
from run_evidence_agent import score, sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args()
    measurements=args.run_dir/'measurements.json';data=json.loads(measurements.read_text())
    rows=['# AVA integration results','',f"Measurement status: {data['status']}. Sources: {len(data['protocol']['sources'])}. Completed clips: {len(data['cases'])}/{data['expected_episodes']}.",'',
          'These are descriptive results from official AVA labels. Timestamps within a clip are correlated. Noise is controlled Gaussian noise (sigma 25), not a claim about real network corruption. The unchanged source video is a diagnostic comparator, not a clean ground-truth image.', '',
          '| Clip type | Candidate | Errors / timestamps | Correct → wrong | Wrong → correct | MUSIQ ↑ | BRISQUE ↓ |',
          '|---|---|---:|---:|---:|---:|---:|']
    for case in data['cases']:
        paired=score(case,{})
        for name,v in case['candidates'].items():
            change=paired['restorers'].get(name,{})
            rows.append(f"| {case['category']} | {name} | {v['error_count']}/{v['n']} | {change.get('harm','—')} | {change.get('corrections','—')} | {v['musiq']:.2f} | {v['brisque']:.2f} |")
    for file in sorted(args.run_dir.glob('agent_*.json')):
        agent=json.loads(file.read_text())
        if agent['identity']['measurements_sha256']!=sha(measurements):raise RuntimeError('Agent results are bound to different measurements')
        rows += ['',f"## {agent['identity']['model']} — {file.name}",'',f"Status: {agent['status']}. Access: measurement tools only; no direct video/audio input.",'',
            '| Clip ID | Status | Selected evidence | Answered / eligible | Agent errors | Capture errors on same frames |',
            '|---|---|---|---:|---:|---:|']
        for c in agent['cases']:
            ev=c['evaluation'];sub=c['run'].get('submission') or {}
            rows.append(f"| {c['id']} | {c['run']['status']} | {sub.get('candidate','—')} | {ev.get('answered',0)}/{ev['eligible']} | {ev.get('agent_errors','—')} | {ev.get('captured_errors_on_same_answered_frames','—')} |")
    rows += ['', '## Interpretation boundaries','',
        'This verifies dataset-to-restoration-to-detector-to-agent execution. It does not establish audio-conditioned hallucinations: FastDVDnet does not take audio. An agent error can reflect the detector evidence supplied to it; it is not automatically a native audiovisual Qwen perception error.', '',
        'The enforced inspect/compare steps are a baseline tool policy, not a validated novel safeguard. One source cannot establish generalization, statistical significance, or a paper-level benefit. Human synchronization review and additional sources remain separate from successful software execution.', '']
    path=args.run_dir/'RESULTS.md';path.write_text('\n'.join(rows),encoding='utf-8');print(path)

if __name__=='__main__':main()

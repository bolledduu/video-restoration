"""Separate tool error propagation from new agent errors using saved real responses."""
import hashlib
import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parents[1]

def main():
    result={'scope':'Reanalysis of actual saved Qwen sessions; no new calls. Report cohorts separately because inputs/prompts differ. Repeated timestamps are dependent.','cohorts':[]}
    for name in ['three_video_qwen_paired','transfer_rq_qwen','benefit_qwen']:
        file=P/'results'/f'{name}.json';d=json.loads(file.read_text());cases={c['id']:c for c in d['cases']};rows=[]
        for r in d['runs']:
            if r['status']!='completed':continue
            c=cases[r['case_id']];condition=r['condition'];tags=np.array(r['submission']['frame_tags'])
            candidate='fastdvdnet' if name=='three_video_qwen_paired' and condition=='restored' else condition
            detector=np.asarray(c['candidates'][candidate]['probability'])>=.5
            if 'scored_labels' in c:y=np.array(c['scored_labels'])
            else:y=np.full(len(tags),c['label'])
            assert len(tags)==len(y)==len(detector)
            answered=tags!=-1;wrong=detector!=y
            counts={'timestamps':len(y),'answered':int(answered.sum()),'tool_errors':int(wrong.sum()),
                'tool_error_repeated':int(np.sum(wrong&answered&(tags!=y))),
                'tool_error_corrected':int(np.sum(wrong&answered&(tags==y))),
                'tool_error_abstained':int(np.sum(wrong&~answered)),
                'agent_new_error_tool_correct':int(np.sum(~wrong&answered&(tags!=y))),
                'agent_abstention_tool_correct':int(np.sum(~wrong&~answered))}
            assert counts['tool_errors']==counts['tool_error_repeated']+counts['tool_error_corrected']+counts['tool_error_abstained']
            assert counts['tool_error_repeated']+counts['agent_new_error_tool_correct']==r['evaluation']['frame_wrong']
            rows.append(dict(case_id=c['id'],condition=condition,repeat=r['repeat'],**counts))
        summary={condition:{k:sum(r[k] for r in rows if r['condition']==condition) for k in counts} for condition in sorted({r['condition'] for r in rows})}
        result['cohorts'].append(dict(name=name,status=d['status'],sessions=len(rows),input_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),conditions=summary,rows=rows))
    (P/'results/agent_propagation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Agent error propagation audit','',result['scope'],'',
        'Qwen receives tool scores, not video/audio. A copied tool error is a pipeline failure but is not evidence that Qwen visually hallucinated. Abstention is recorded separately from correction.','']
    for cohort in result['cohorts']:
        lines += [f"## {cohort['name']} ({cohort['sessions']} saved sessions; {cohort['status']})",'',
            '| Input condition | Answered / timestamps | Tool errors | Repeated by agent | Corrected by agent | Agent abstained on tool errors | New agent errors on tool-correct timestamps |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name,s in cohort['conditions'].items():lines.append(f"| {name} | {s['answered']}/{s['timestamps']} | {s['tool_errors']} | {s['tool_error_repeated']} | {s['tool_error_corrected']} | {s['tool_error_abstained']} | {s['agent_new_error_tool_correct']} |")
    lines += ['', 'The benefit cohort is incomplete; its saved sessions are not an unbiased estimate of all selected cases. No missing session is imputed. This audit diagnoses the existing interface and is not independent confirmation of a new agent or method.','']
    (P/'AGENT_PROPAGATION_AUDIT_2026-10-16.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps([{k:c[k] for k in ['name','sessions','conditions']} for c in result['cohorts']],indent=2))

if __name__=='__main__':main()

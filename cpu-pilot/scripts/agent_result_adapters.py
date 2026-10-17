"""Versioned adapters for historical agent formats; missing scores are never zero."""
from run_evidence_agent import score


def legacy_runs(document, cohort):
    cases={c['id']:c for c in document['cases']};rows=[]
    for run in document['runs']:
        if run['status'].lower() not in ('complete','completed'):
            rows.append(dict(cohort=cohort,clip_id=run['case_id'],status=run['status'],evaluation=None));continue
        case=cases[run['case_id']]
        n=len(case['candidates']['captured']['probability'])
        tags=run['submission']['frame_tags']
        if len(tags)!=n or any(type(v) is not int or v not in (-1,0,1) for v in tags):
            raise ValueError('Malformed legacy frame tags')
        truth=case['labels'] if 'labels' in case else [case['label']]*n
        converted={'labels':truth,'candidates':case['candidates']}
        decisions=[{-1:'abstain',0:'not_speaking',1:'speaking'}[v] for v in tags]
        ev=score(converted,{'submission':{'decisions':decisions}})
        if ev['agent_errors']!=run['evaluation']['frame_wrong']:
            raise ValueError('Stored and recomputed error counts disagree')
        rows.append(dict(cohort=cohort,clip_id=run['case_id'],status='COMPLETE',
            model=document['protocol']['model'],condition=run['condition'],repeat=run['repeat'],evaluation=ev))
    return rows

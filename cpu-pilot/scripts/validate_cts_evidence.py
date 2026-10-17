"""Validate measured audit provenance and prevent unsupported final-result claims."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from evaluate_captured_support import transition_counts,audit_grid_transition_counts

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--training-measurements',type=Path,nargs='*',default=[]);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    data=json.loads(args.audit.read_text());ev=json.loads(args.evaluation.read_text());protocol_path=args.audit.with_name('protocol.json');protocol=json.loads(protocol_path.read_text())
    if data['status']!='COMPLETE' or data['protocol_sha256']!=sha(protocol_path):raise ValueError('Audit incomplete or provenance changed')
    if ev['input_sha256']!=sha(args.audit):raise ValueError('Evaluation input mismatch')
    if protocol['core_sha256']!=sha(Path(__file__).with_name('captured_support.py')):raise ValueError('Audit implementation changed')
    cases=data['cases'];full=transition_counts(cases);central=audit_grid_transition_counts(cases);states=['CC','CI_harm','IC_correction','II']
    sources_by_class=defaultdict(set)
    for c in cases:
        for state in set(c['transition_targets']):sources_by_class[states[state]].add(c['source'])
        if len(c['folds'])!=2:raise ValueError('Missing audit fold')
        for f in c['folds']:
            if len(f['masked_input_sha256'])!=64 or len(f['audit_output_sha256'])!=64:raise ValueError('Missing real masked execution receipt')
            for name in ('lower_face','cheek','full_frame'):
                region=f['regions'][name]
                if not region['indices'] or not np.isfinite(region['support']).all():raise ValueError('Missing temporal support measurement')
                if any(s not in region['probe_probabilities'] for s in ('shift_-3','shift_3')):raise ValueError('Missing detector control')
    methods={}
    for variant in ('base','ordinary_audit','support'):
        supported=[f for f in ev['folds'] if f['variants'][variant]['status']=='EVALUATED']
        rows=[r for f in supported for r in f['variants'][variant]['rows']]
        ids={r['case_id'] for r in rows};baseline={name:[r for r in values if r['case_id'] in ids] for name,values in ev['baselines'].items()}
        summary=lambda values:{k:sum(r[k] for r in values) for k in ('eligible','errors','captured_errors','harm','corrections')}
        methods[variant]=dict(evaluable_sources=[f['test_source'] for f in supported],evaluated=summary(rows),matched_baselines={name:summary(values) for name,values in baseline.items()},
            selected_restored_cases=sum(r['selected']=='restored' for r in rows),blocked_sources=[f['test_source'] for f in ev['folds'] if f['variants'][variant]['status']!='EVALUATED'])
    new_training=[]
    for file in args.training_measurements:
        d=json.loads(file.read_text())
        if d['status'] not in ('COMPLETE','COMPLETE_WITH_EXCLUSIONS'):raise ValueError('Training measurement incomplete: '+str(file))
        for c in d['cases']:
            y=np.asarray(c['labels']);base=np.asarray(c['candidates']['captured']['probability'])>=.5
            for name,v in c['candidates'].items():
                if name in ('captured','reference'):continue
                pred=np.asarray(v['probability'])>=.5
                new_training.append(dict(case_id=c['id'],source=c['source'],restorer=name,eligible=len(y),harm=int(np.sum((base==y)&(pred!=y))),corrections=int(np.sum((base!=y)&(pred==y)))))
    blockers=[]
    if len(sources_by_class['IC_correction'])<2:blockers.append('Correction examples in the measured audit cohort occur on fewer than two sources')
    if methods['support']['blocked_sources']:blockers.append(str(len(methods['support']['blocked_sources']))+' source folds lack required training transitions after source separation')
    matched=methods['support']['matched_baselines'].get('restored',{})
    if not matched.get('harm',0) and not matched.get('corrections',0):blockers.append('Evaluable folds contain no restoration-induced decision transitions')
    blockers.extend(['These development sources were previously inspected; they are not fresh confirmation','This report includes no human-preference validation','This run does not execute the frozen final calibration/test protocol'])
    result=dict(status='DEVELOPMENT_EVALUATION_COMPLETE',final_method_validation='NOT_ESTABLISHED',
        audit_sha256=sha(args.audit),evaluation_sha256=sha(args.evaluation),actual_masked_cases=len(cases),actual_sources=len({c['source'] for c in cases}),
        full_grid_transition_counts=dict(zip(states,full)),audit_grid_transition_counts=dict(zip(states,central)),transition_sources={k:sorted(v) for k,v in sources_by_class.items()},
        methods=methods,new_training_pairs=new_training,
        blockers=blockers,
        retained_final_data='Frozen AVA protocol has 5 train, 5 calibration, 15 test sources; no calibration/test inference executed here')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('Actual audit source/class support:',result['transition_sources'])
    print('Measured full-grid transitions:',result['full_grid_transition_counts'])
    print('Final validation:',result['final_method_validation'])
if __name__=='__main__':main()

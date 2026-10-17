"""Source-separated development assessment of measured captured-support features.

Historical source folds are exploratory. No final-test success is implied.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,roc_auc_score
from captured_support import logits,transition_targets

BASE=['captured_logit','restored_logit','signed_score_change','threshold_disagreement','confidence_gain','delta_musiq','delta_brisque']
AUDIT=['heldout_mse','full_audit_logit_instability','full_audit_threshold_instability']
SUPPORT=['lower_face_timing_support','mask_support_std','signed_timing_influence','cheek_timing_support','fullframe_timing_support','static_influence','half_amplitude_influence','onehalf_amplitude_influence']

def extract(case):
    # Deliberately does not read labels, transition targets or reference imagery.
    indices=np.asarray(case['indices']);pc=np.asarray(case['captured_probability']);pr=np.asarray(case['restored_probability'])
    eligible=set(indices.tolist())
    for fold in case['folds']:eligible &= set(fold['regions']['lower_face']['indices'])
    times=np.array(sorted(eligible));pos=np.array([int(np.flatnonzero(indices==t)[0]) for t in times])
    if not len(times):raise ValueError('No common factual/control support')
    quality=case['quality'];q=quality[case['candidate_key']];c=quality['captured']
    base=np.column_stack([logits(pc[pos]),logits(pr[pos]),pr[pos]-pc[pos],(pr[pos]>=.5)!=(pc[pos]>=.5),
        abs(pr[pos]-.5)-abs(pc[pos]-.5),np.full(len(pos),q['musiq']-c['musiq']),np.full(len(pos),c['brisque']-q['brisque'])])
    folds=[]
    for f in case['folds']:
        factual=np.asarray(f['audit_probability'])[pos]
        def region_values(name,key):
            r=f['regions'][name];j=[r['indices'].index(int(t)) for t in times]
            return np.asarray(r[key])[j]
        mouth=f['regions']['lower_face'];mi=[mouth['indices'].index(int(t)) for t in times]
        shifted=np.mean([logits(np.asarray(mouth['probe_probabilities']['shift_'+str(s)])[pos]) for s in (-3,3)],axis=0)
        folds.append(dict(mse=np.asarray(mouth['mse']['factual'])[mi],stability=abs(logits(pr[pos])-logits(factual)),
            flips=(pr[pos]>=.5)!=(factual>=.5),support=region_values('lower_face','support'),influence=logits(factual)-shifted,
            cheek=region_values('cheek','support'),full=region_values('full_frame','support'),
            static=logits(factual)-logits(np.asarray(mouth['probe_probabilities']['static'])[pos]),
            half=logits(factual)-logits(np.asarray(mouth['probe_probabilities']['scale_0.5'])[pos]),
            onehalf=logits(factual)-logits(np.asarray(mouth['probe_probabilities']['scale_1.5'])[pos])))
    mean=lambda key:np.mean([f[key] for f in folds],axis=0)
    audit=np.column_stack([mean('mse'),mean('stability'),mean('flips')])
    support=np.column_stack([mean('support'),np.std([f['support'] for f in folds],axis=0),mean('influence'),mean('cheek'),mean('full'),mean('static'),mean('half'),mean('onehalf')])
    x=np.column_stack([base,audit,support])
    if not np.isfinite(x).all():raise ValueError('Nonfinite diagnostic')
    return x,pos

def transition_counts(cases):
    total=np.zeros(4,int)
    for c in cases:total+=np.bincount(transition_targets(c['labels'],c['captured_probability'],c['restored_probability']),minlength=4)
    return total.tolist()

def audit_grid_transition_counts(cases):
    total=np.zeros(4,int)
    for c in cases:
        _,pos=extract(c)
        total+=np.bincount(transition_targets(np.asarray(c['labels'])[pos],np.asarray(c['captured_probability'])[pos],np.asarray(c['restored_probability'])[pos]),minlength=4)
    return total.tolist()

def fit(cases,variant):
    features=[];targets=[];sources=[]
    for c in cases:
        x,pos=extract(c);features.append(x);targets.extend(transition_targets(np.asarray(c['labels'])[pos],np.asarray(c['captured_probability'])[pos],np.asarray(c['restored_probability'])[pos]).tolist());sources.extend([c['source']]*len(pos))
    y=np.asarray(targets)
    if not {0,1,2,3}<=set(y.tolist()):raise ValueError('Training data lack one or more CC/CI/IC/II transitions')
    width={'base':len(BASE),'ordinary_audit':len(BASE)+len(AUDIT),'support':len(BASE)+len(AUDIT)+len(SUPPORT)}[variant]
    x=np.concatenate(features)[:,:width];scaler=StandardScaler().fit(x)
    weights=np.array([1/sources.count(s) for s in sources]);weights*=len(weights)/weights.sum()
    classifier=LogisticRegression(C=1.,max_iter=2000,random_state=20261017)
    classifier.fit(scaler.transform(x),y,sample_weight=weights)
    return dict(variant=variant,width=width,scaler=scaler,classifier=classifier)

def predict(model,case):
    x,pos=extract(case);probs=model['classifier'].predict_proba(model['scaler'].transform(x[:,:model['width']]))
    if list(model['classifier'].classes_)!=[0,1,2,3]:raise ValueError('Incomplete transition classifier')
    return probs,pos

def evaluate(case,restore):
    y=np.asarray(case['labels']);c=np.asarray(case['captured_probability'])>=.5;r=np.asarray(case['restored_probability'])>=.5;p=r if restore else c
    q=case['quality'];delta=q[case['candidate_key']]['musiq']-q['captured']['musiq']
    return dict(case_id=case['id'],source=case['source'],eligible=len(y),errors=int(np.sum(p!=y)),captured_errors=int(np.sum(c!=y)),
        harm=int(np.sum((c==y)&(p!=y))),corrections=int(np.sum((c!=y)&(p==y))),selected='restored' if restore else 'captured',delta_musiq=delta if restore else 0.)

def choose(model,case,tau):
    probs,pos=predict(model,case);harm=float(probs[:,1].mean());correction=float(probs[:,2].mean())
    quality=case['quality'][case['candidate_key']]['musiq']>case['quality']['captured']['musiq']
    return bool(quality and harm<=tau and correction>=harm),dict(estimated_harm=harm,estimated_correction=correction,audited_timestamps=len(pos))

def calibrate(model,cases):
    candidates=[]
    for tau in (0.,.01,.02,.05,.1,.2,.5,1.):
        rows=[evaluate(c,choose(model,c,tau)[0]) for c in cases]
        h=sum(r['harm'] for r in rows);gain=sum(r['corrections']-r['harm'] for r in rows);quality=sum(r['delta_musiq'] for r in rows)
        candidates.append(dict(tau=tau,harm=h,net_gain=gain,quality=quality))
    feasible=[r for r in candidates if r['harm']==0 and r['net_gain']>=0]
    best=max(feasible,key=lambda r:(r['net_gain'],r['quality'],-r['tau']))
    return best['tau'],candidates

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--measurements',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    d=json.loads(args.measurements.read_text())
    if d['status']!='COMPLETE':raise ValueError('Measurements incomplete')
    cases=d['cases'];sources=sorted({c['source'] for c in cases});folds=[]
    for i,test_source in enumerate(sources):
        cal_source=sources[(i+1)%len(sources)];train=[c for c in cases if c['source'] not in (test_source,cal_source)]
        cal=[c for c in cases if c['source']==cal_source];test=[c for c in cases if c['source']==test_source]
        row=dict(test_source=test_source,calibration_source=cal_source,training_sources=sorted({c['source'] for c in train}),training_transitions=transition_counts(train),training_audit_grid_transitions=audit_grid_transition_counts(train),calibration_transitions=transition_counts(cal),variants={})
        for variant in ('base','ordinary_audit','support'):
            try:model=fit(train,variant)
            except ValueError as error:row['variants'][variant]=dict(status='INSUFFICIENT_TRAINING_SUPPORT',reason=str(error));continue
            tau,grid=calibrate(model,cal);evaluated=[];y=[];risk=[]
            for c in test:
                selected,diagnostic=choose(model,c,tau);evaluated.append({**evaluate(c,selected),**diagnostic})
                probs,pos=predict(model,c);target=transition_targets(np.asarray(c['labels'])[pos],np.asarray(c['captured_probability'])[pos],np.asarray(c['restored_probability'])[pos]);y.extend((target==1).astype(int).tolist());risk.extend(probs[:,1].tolist())
            auc=float(roc_auc_score(y,risk)) if len(set(y))==2 else None
            ap_score=float(average_precision_score(y,risk)) if len(set(y))==2 else None
            row['variants'][variant]=dict(status='EVALUATED',tau=tau,calibration_grid=grid,rows=evaluated,harm_auroc=auc,harm_ap=ap_score,
                model=dict(classes=model['classifier'].classes_.tolist(),coef=model['classifier'].coef_.tolist(),intercept=model['classifier'].intercept_.tolist(),mean=model['scaler'].mean_.tolist(),scale=model['scaler'].scale_.tolist()))
        folds.append(row)
    out=dict(scope='Exploratory leave-one-source-out development assessment; a separate source calibrates each fold. These are previously inspected sources, NOT final validation.',
        input_sha256=hashlib.sha256(args.measurements.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),features=BASE+AUDIT+SUPPORT,
        selection='Whole-clip MUSIQ improvement, estimated correction >= estimated harm; harm threshold calibrated independently with zero observed calibration harm; no population risk guarantee',
        fold_assignment='Sorted source order; next source cyclically is calibration, all remaining sources train; fixed before reading scores',folds=folds,
        baselines={name:[evaluate(c, name=='restored' or (name=='musiq' and c['quality'][c['candidate_key']]['musiq']>c['quality']['captured']['musiq'])) for c in cases] for name in ('captured','restored','musiq')})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2,allow_nan=False))
    for variant in ('base','ordinary_audit','support'):
        rows=[r for f in folds if f['variants'][variant]['status']=='EVALUATED' for r in f['variants'][variant]['rows']]
        print(variant,'evaluable sources',len({r['source'] for r in rows}),{k:sum(r[k] for r in rows) for k in ('eligible','errors','captured_errors','harm','corrections')})
if __name__=='__main__':main()

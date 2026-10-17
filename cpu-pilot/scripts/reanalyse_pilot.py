"""Recompute descriptive policy counts; never refit a safeguard on pilot test outcomes."""
import argparse, json
from pathlib import Path
from run_final_experiment import load_all_experimental_data, evaluate_policy_on_pair
from build_full_benchmark import build_cases

POLICIES=('captured_only','restored_only','quality_musiq','quality_brisque',
          'quality_consensus','confidence_max','disagreement_capture_fallback',
          'disagreement_abstain','oracle_upper_bound')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    dev,pairs=load_all_experimental_data()
    cases=build_cases(pairs) # Fail closed on pairing conflicts before scoring anything.
    results={}
    for policy in POLICIES:
        rows=[evaluate_policy_on_pair(policy,p) for p in pairs]
        answered=sum(r['answered_frames'] for r in rows)
        errors=sum(r['errors'] for r in rows)
        results[policy]=dict(eligible=sum(r['eligible_frames'] for r in rows),answered=answered,
            errors=errors,error_rate=errors/answered if answered else None,
            captured_errors_same_answered=sum(r['captured_errors_on_answered'] for r in rows),
            harm=sum(r['harm'] for r in rows),corrections=sum(r['corrections'] for r in rows),
            quality_available_for_all=all(r['quality_available'] for r in rows),rows=rows)
    report=dict(scope='Historical exploratory pairs; repeated timestamps across restorers, not independent observations',
        method_evaluated='Fixed baseline selectors only; no proposed CTS framework',
        source_count=len({p['source'] for p in pairs}),pair_count=len(pairs),episode_count=len(cases),
        policies=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    for k,v in results.items():print(k,{a:b for a,b in v.items() if a!='rows'})
if __name__=='__main__':main()

"""Continue only the available model; preserve original interrupted pilot and responses."""
import argparse,hashlib,json,time
from pathlib import Path
from run_gemini_av_pilot import Gemini,sha,save,make_payload,score_answer,validate_answer,CONFIG,PROMPT,paired_counts

MODEL='gemini-3.5-flash-lite'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    p=args.output/'protocol.json';protocol=json.loads(p.read_text())
    if protocol['code_sha256']!=sha(Path(__file__).with_name('run_gemini_av_pilot.py')):raise ValueError('Original frozen implementation changed')
    if protocol['prompt']!=PROMPT or protocol['generation_config']!=CONFIG:raise ValueError('Frozen inference configuration changed')
    original=args.output/'runs.json';old=json.loads(original.read_text());target=args.output/'runs_gemini_3_5.json'
    if target.exists():raise FileExistsError('Preserve attempts; no automatic rerun')
    selected=[r for r in old['runs'] if r['model']==MODEL]
    if any(r['status']!='COMPLETE' for r in selected):raise ValueError('Existing incomplete calls for this model require separate review')
    result=dict(status='RUNNING',protocol_sha256=sha(p),continuation_script_sha256=sha(__file__),inherited_runs_sha256=sha(original),
        unavailable_model='gemini-2.5-flash-lite',unavailability='Google NOT_FOUND: model no longer available to new users; countTokens and generateContent both rejected',
        reserved_upper_cost_usd=sum(r['reserved_upper_cost_usd'] for r in selected),runs=selected)
    save(target,result);api=Gemini();cases={c['id']:c for c in protocol['cases']}
    done={(r['case_id'],r['condition'],r['repeat']) for r in selected};counts={}
    for job in protocol['jobs']:
        if job['model']!=MODEL or (job['case_id'],job['condition'],job['repeat']) in done:continue
        case=cases[job['case_id']];media=case['media'][job['condition']]
        if sha(media['path'])!=media['sha256']:raise ValueError('Media hash changed')
        payload=make_payload(media['path']);row={**job,'media_sha256':media['sha256'],'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
        try:
            if media['sha256'] not in counts:counts[media['sha256']]=api.request('POST','models/'+MODEL+':countTokens',{'contents':payload['contents']})['totalTokens']
            count=counts[media['sha256']];reserve=count*.30/1e6+1024*2.50/1e6
            if count>50000 or result['reserved_upper_cost_usd']+reserve>protocol['cost_limit_usd']:raise RuntimeError('Cost/token ceiling reached')
            row.update(status='REQUEST_PENDING',input_token_count=count,reserved_upper_cost_usd=reserve)
            result['reserved_upper_cost_usd']+=reserve;result['runs'].append(row);save(target,result)
            response=api.request('POST','models/'+MODEL+':generateContent',payload);row['response']=response
            candidates=response.get('candidates',[])
            if len(candidates)!=1 or candidates[0].get('finishReason')!='STOP':raise ValueError('Response did not complete normally')
            text=''.join(part.get('text','') for part in candidates[0]['content']['parts'] if not part.get('thought'))
            answer=validate_answer(json.loads(text));row.update(status='COMPLETE',answer=answer,evaluation=score_answer(answer,case['truth'],case['majority_truth']))
            save(target,result);print(job['case_id'],job['condition'],job['repeat'],row['evaluation'],flush=True)
        except (RuntimeError,ValueError,KeyError) as error:
            row.update(status='FAILED',error=str(error))
            if not result['runs'] or result['runs'][-1] is not row:result['runs'].append(row)
            result['status']='INCOMPLETE';save(target,result);raise SystemExit('Stopped: '+str(error)) from None
        time.sleep(3)
    result['status']='COMPLETE_FOR_AVAILABLE_MODEL';save(target,result)
    rows=[]
    for repeat in range(2):
        for cid,case in cases.items():
            pair={r['condition']:r for r in result['runs'] if r['repeat']==repeat and r['case_id']==cid}
            if set(pair)!={'captured','fastdvdnet'} or any(r['status']!='COMPLETE' for r in pair.values()):raise ValueError('Incomplete matched pair')
            c=pair['captured']['answer'];r=pair['fastdvdnet']['answer']
            rows.append(dict(case_id=cid,source=case['source'],repeat=repeat,truth=case['truth'],majority_truth=case['majority_truth'],captured=c,restored=r,
                **paired_counts(c,r,case['truth']),majority_captured_error=int(c['majority_speaking']!=case['majority_truth']),majority_restored_error=int(r['majority_speaking']!=case['majority_truth'])))
    summary=dict(model=MODEL,completed_requests=len(result['runs']),source_count=len({c['source'] for c in cases.values()}),paired_rows=rows,
        totals={k:sum(r[k] for r in rows) for k in ('jointly_answered','captured_errors','restored_errors','harm','corrections','majority_captured_error','majority_restored_error')},
        note='Four development clips, two sources, two repetitions; repeated bins are not independent samples. No method or agent-tool evaluation.',unavailable_model='gemini-2.5-flash-lite')
    save(args.output/'paired_summary_gemini_3_5.json',summary);print(summary['totals'],flush=True)
if __name__=='__main__':main()

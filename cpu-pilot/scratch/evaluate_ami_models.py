import json

models = [
    ('qwen/qwen3.8-27b', 'results/ami_v1/agent_groq_qwen_qwen3_8-27b_interval_v2.json'),
    ('openai/gpt-oss-20b', 'results/ami_v1/agent_groq_gpt_oss_20b_interval_v1.json'),
    ('openai/gpt-oss-120b', 'results/ami_v1/agent_groq_gpt_oss_120b_interval_v1.json')
]

for model_name, path in models:
    data = json.load(open(path))
    print(f'=== {model_name} on AMI Meeting Corpus ===')
    print('Status:', data['status'])
    tot_ans = sum(c['evaluation']['answered'] for c in data['cases'])
    tot_elig = sum(c['evaluation']['eligible'] for c in data['cases'])
    tot_agent_err = sum(c['evaluation']['agent_errors'] for c in data['cases'])
    tot_cap_err = sum(c['evaluation']['captured_errors'] for c in data['cases'])
    print(f'Coverage: {tot_ans}/{tot_elig} ({tot_ans/tot_elig*100:.1f}%)')
    print(f'Captured Errors: {tot_cap_err}, Agent Errors: {tot_agent_err}')
    for c in data['cases']:
        print(f"  {c['id']}:")
        print(f"    Candidate: {c['run']['submission']['candidate']}")
        print(f"    Agent Errors: {c['evaluation']['agent_errors']} / Captured Errors: {c['evaluation']['captured_errors']}")
        print(f"    Rationale: {c['run']['submission']['rationale'][:120]}")
    print()

import json
import sys

def show_rationales(title, files):
    print(f"\n==================== {title} ====================")
    for m, f in files:
        d = json.load(open(f, encoding='utf-8'))
        print(f"\n--- Model: {m} ---")
        for i, c in enumerate(d['cases']):
            cand = c['run']['submission']['candidate']
            rat = c['run']['submission']['rationale'].encode('ascii', errors='backslashreplace').decode('ascii')
            print(f"  Case {i+1} [{c['id']}]: Candidate={cand}")
            print(f"    Rationale: {rat}\n")

show_rationales("AMI Meeting Corpus", [
    ("qwen/qwen3.8-27b", "cpu-pilot/results/ami_v1/agent_groq_qwen_qwen3_8-27b_interval_v2.json"),
    ("openai/gpt-oss-20b", "cpu-pilot/results/ami_v1/agent_groq_gpt_oss_20b_interval_v1.json"),
    ("openai/gpt-oss-120b", "cpu-pilot/results/ami_v1/agent_groq_gpt_oss_120b_interval_v1.json"),
])

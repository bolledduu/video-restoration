import json
import glob
import os

files_to_check = {
    "AVA-ActiveSpeaker": [
        ("qwen/qwen3.8-27b", "cpu-pilot/results/ava_smoke_v1/agent_groq_qwen_qwen3_8-27b_interval_v1.json"),
        ("openai/gpt-oss-20b", "cpu-pilot/results/ava_smoke_v1/agent_groq_gpt_oss_20b_interval_v6.json"),
        ("openai/gpt-oss-120b", "cpu-pilot/results/ava_smoke_v1/agent_groq_gpt_oss_120b_interval_v1.json"),
    ],
    "AMI Meeting Corpus": [
        ("qwen/qwen3.8-27b", "cpu-pilot/results/ami_v1/agent_groq_qwen_qwen3_8-27b_interval_v2.json"),
        ("openai/gpt-oss-20b", "cpu-pilot/results/ami_v1/agent_groq_gpt_oss_20b_interval_v1.json"),
        ("openai/gpt-oss-120b", "cpu-pilot/results/ami_v1/agent_groq_gpt_oss_120b_interval_v1.json"),
    ]
}

for ds, models in files_to_check.items():
    print(f"\n==================== {ds} ====================")
    for model_name, fpath in models:
        if not os.path.exists(fpath):
            print(f"Missing file: {fpath}")
            continue
        with open(fpath, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        
        cases = data.get("cases", [])
        total_eligible = sum(c.get("evaluation", {}).get("eligible", 0) for c in cases)
        total_answered = sum(c.get("evaluation", {}).get("answered", 0) for c in cases)
        total_cap_err = sum(c.get("evaluation", {}).get("captured_errors", 0) for c in cases)
        total_agent_err = sum(c.get("evaluation", {}).get("agent_errors", 0) for c in cases)
        
        tool_counts = [len(c.get("run", {}).get("trace", [])) for c in cases]
        candidates = [c.get("run", {}).get("submission", {}).get("candidate", "none") for c in cases]
        
        print(f"\nModel: {model_name}")
        print(f"  File: {fpath}")
        print(f"  Episodes/Cases: {len(cases)}")
        print(f"  Total Timestamps: {total_eligible}")
        print(f"  Coverage: {total_answered}/{total_eligible} ({total_answered/total_eligible*100:.1f}%)")
        print(f"  Baseline Captured Errors: {total_cap_err} ({total_cap_err/total_eligible*100:.2f}%)")
        print(f"  Agent Final Errors:       {total_agent_err} ({total_agent_err/total_eligible*100:.2f}%)")
        print(f"  Net Error Reduction:      {total_cap_err - total_agent_err} frames")
        print(f"  Candidates Selected:      {candidates}")
        print(f"  Trace Turns per Case:     {tool_counts}")
        
        for i, c in enumerate(cases):
            cid = c.get("id")
            cand = c.get("run", {}).get("submission", {}).get("candidate")
            cev = c.get("evaluation", {})
            print(f"    Case {i+1} [{cid}]: candidate={cand}, cap_err={cev.get('captured_errors')}, ag_err={cev.get('agent_errors')}")

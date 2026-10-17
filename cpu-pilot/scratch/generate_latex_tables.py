import json
from pathlib import Path

# Files
ava_files = {
    "Qwen3.8-27B": "cpu-pilot/results/ava_smoke_v1/agent_groq_qwen_qwen3_8-27b_interval_v1.json",
    "GPT-OSS-20B": "cpu-pilot/results/ava_smoke_v1/agent_groq_gpt_oss_20b_interval_v6.json",
    "GPT-OSS-120B": "cpu-pilot/results/ava_smoke_v1/agent_groq_gpt_oss_120b_interval_v1.json",
}

ami_files = {
    "Qwen3.8-27B": "cpu-pilot/results/ami_v1/agent_groq_qwen_qwen3_8-27b_interval_v2.json",
    "GPT-OSS-20B": "cpu-pilot/results/ami_v1/agent_groq_gpt_oss_20b_interval_v1.json",
    "GPT-OSS-120B": "cpu-pilot/results/ami_v1/agent_groq_gpt_oss_120b_interval_v1.json",
}

def get_stats(fpath):
    d = json.load(open(fpath, encoding='utf-8'))
    cases = d['cases']
    tot_ans = sum(c['evaluation']['answered'] for c in cases)
    tot_elig = sum(c['evaluation']['eligible'] for c in cases)
    tot_cap_err = sum(c['evaluation']['captured_errors'] for c in cases)
    tot_ag_err = sum(c['evaluation']['agent_errors'] for c in cases)
    cap_rate = (tot_cap_err / tot_elig) * 100
    ag_rate = (tot_ag_err / tot_ans) * 100
    cov = (tot_ans / tot_elig) * 100
    return {
        "eligible": tot_elig,
        "answered": tot_ans,
        "cap_err": tot_cap_err,
        "ag_err": tot_ag_err,
        "cap_rate": cap_rate,
        "ag_rate": ag_rate,
        "coverage": cov
    }

print("% =================================================================")
print("% Table 1: Cross-Model Transfer (LaTeX formatted)")
print("% =================================================================")
print(r"\begin{table}[t]")
print(r"\centering\small")
print(r"\caption{Cross-model evaluation across core datasets under TalkNet detector. Format: Captured Error / Restored Agent Error (\%) at 100\% answer coverage.}")
print(r"\label{tab:modeltransfer}")
print(r"\setlength{\tabcolsep}{4pt}")
print(r"\begin{tabular}{@{}llcc@{}}")
print(r"\toprule")
print(r"Controller & Detector & AVA-ActiveSpeaker & AMI Meeting Corpus \\")
print(r"\midrule")

for model in ["Qwen3.8-27B", "GPT-OSS-20B", "GPT-OSS-120B"]:
    ava_s = get_stats(ava_files[model])
    ami_s = get_stats(ami_files[model])
    
    ava_str = f"{ava_s['cap_rate']:.1f} / {ava_s['ag_rate']:.1f}\\%"
    ami_str = f"{ami_s['cap_rate']:.1f} / {ami_s['ag_rate']:.1f}\\%"
    print(f"{model:14s} & TalkNet & {ava_str} & {ami_str} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
print("\n")

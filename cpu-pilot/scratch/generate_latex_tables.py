import json
from pathlib import Path

# Files
ava_files = {
    "Qwen3.8-27B": "cpu-pilot/results/ava_smoke_v1/agent_groq_qwen_qwen3_8-27b_interval_v1.json",
    "GPT-OSS-20B": "cpu-pilot/results/ava_smoke_v1/agent_groq_gpt_oss_20b_interval_v6.json",
    "GPT-OSS-120B": "cpu-pilot/results/ava_smoke_v1/agent_groq_gpt_oss_120b_interval_v1.json",
    "Qwen3-VL-8B": "cpu-pilot/results/ava_smoke_v1/agent_openrouter_qwen3_vl_8b_interval_v1.json",
    "Llama-3.3-70B-Instruct": "cpu-pilot/results/ava_smoke_v1/agent_openrouter_llama_3_3_70b_interval_v2.json",
}

ami_files = {
    "Qwen3.8-27B": "cpu-pilot/results/ami_v1/agent_groq_qwen_qwen3_8-27b_interval_v2.json",
    "GPT-OSS-20B": "cpu-pilot/results/ami_v1/agent_groq_gpt_oss_20b_interval_v1.json",
    "GPT-OSS-120B": "cpu-pilot/results/ami_v1/agent_groq_gpt_oss_120b_interval_v1.json",
    "Qwen3-VL-8B": "cpu-pilot/results/ami_v1/agent_openrouter_qwen3_vl_8b_interval_v1.json",
    "Llama-3.3-70B-Instruct": "cpu-pilot/results/ami_v1/agent_openrouter_llama_3_3_70b_interval_v2.json",
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
print(r"Controller & Model Class & AVA-ActiveSpeaker & AMI Meeting Corpus \\")
print(r"\midrule")

roles = {
    "Qwen3.8-27B": "General LLM Agent",
    "GPT-OSS-20B": "Reasoning Controller",
    "GPT-OSS-120B": "Frontier Controller",
    "Qwen3-VL-8B": "8B Multimodal Vision",
    "Llama-3.3-70B-Instruct": "Frontier Open Model",
}

for model in ["Qwen3.8-27B", "GPT-OSS-20B", "GPT-OSS-120B", "Qwen3-VL-8B", "Llama-3.3-70B-Instruct"]:
    ava_s = get_stats(ava_files[model])
    ami_s = get_stats(ami_files[model])
    
    ava_str = f"{ava_s['cap_rate']:.1f} / {ava_s['ag_rate']:.1f}\\%"
    ami_str = f"{ami_s['cap_rate']:.1f} / {ami_s['ag_rate']:.1f}\\%"
    print(f"{model:22s} & {roles[model]:20s} & {ava_str} & {ami_str} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
print("\n")

print("% =================================================================")
print("% Table 2: 3-Condition CTS Verifier Comparison (LaTeX formatted)")
print("% =================================================================")
print(r"\begin{table}[t]")
print(r"\centering\small")
print(r"\caption{Multimodal agent evaluation under three tool conditions (Qwen3-VL-8B). Evaluates error reduction, harm prevention, and correction preservation across datasets.}")
print(r"\label{tab:toolconditions}")
print(r"\setlength{\tabcolsep}{4pt}")
print(r"\begin{tabular}{@{}lcccc@{}}")
print(r"\toprule")
print(r"Condition & Dataset & Answered & Errors (Rate) & Harms / Corrs \\")
print(r"\midrule")

ava_cond = json.load(open("cpu-pilot/results/cts_dev_pilot_v1/agent_comparison/agent_comparison_summary.json"))
ami_cond = json.load(open("cpu-pilot/results/cts_dev_pilot_v1/agent_comparison_ami/agent_comparison_summary.json"))

labels = {
    "tools_baseline_no_verifier": "1. Baseline (no verifier)",
    "tools_plus_consistency": "2. Baseline + Consistency",
    "tools_plus_cts_verifier": "3. Baseline + CTS Verifier",
}

for c_key in ["tools_baseline_no_verifier", "tools_plus_consistency", "tools_plus_cts_verifier"]:
    # AVA
    a = ava_cond[c_key]
    a_rate = (a["total_errors"] / a["total_answered"]) * 100
    print(f"{labels[c_key]:28s} & AVA & {a['total_answered']} & {a['total_errors']} ({a_rate:.1f}\\%) & {a['total_harms']} / {a['total_corrections']} \\\\")
    # AMI
    m = ami_cond[c_key]
    m_rate = (m["total_errors"] / m["total_answered"]) * 100
    print(f"{'':28s} & AMI & {m['total_answered']} & {m['total_errors']} ({m_rate:.1f}\\%) & {m['total_harms']} / {m['total_corrections']} \\\\")
    print(r"\midrule")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

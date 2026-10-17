import json
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "cts_dev_pilot_v1"

# Model evaluation files (AVA & AMI)
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

# Load 3-condition summaries across all 4 datasets
summaries = {
    "AVA-ActiveSpeaker": json.load(open("cpu-pilot/results/cts_dev_pilot_v1/agent_comparison/ava/ava_agent_comparison_summary.json")),
    "AMI-Meeting": json.load(open("cpu-pilot/results/cts_dev_pilot_v1/agent_comparison/ami/ami_agent_comparison_summary.json")),
    "UniTalk-ASD": json.load(open("cpu-pilot/results/cts_dev_pilot_v1/agent_comparison/unitalk/unitalk_agent_comparison_summary.json")),
    "OmniVR-Benchmark": json.load(open("cpu-pilot/results/cts_dev_pilot_v1/agent_comparison/omnivr/omnivr_agent_comparison_summary.json")),
}

print("% =================================================================")
print("% Table 2: 3-Condition CTS Verifier Comparison (All 4 Datasets)")
print("% =================================================================")
print(r"\begin{table*}[t]")
print(r"\centering\small")
print(r"\caption{Multimodal agent evaluation under three tool conditions across all four datasets (Qwen3-VL-8B). Evaluates error reduction, harm prevention, and correction preservation under matched turn budget.}")
print(r"\label{tab:toolconditions_alldatasets}")
print(r"\setlength{\tabcolsep}{5pt}")
print(r"\begin{tabular}{@{}llcccc@{}}")
print(r"\toprule")
print(r"Condition & Dataset & Answered & Errors (Rate) & Harms / Corrs & Net Impact \\")
print(r"\midrule")

c_labels = [
    ("tools_baseline_no_verifier", "1. Baseline (No Verifier)"),
    ("tools_plus_consistency", "2. Baseline + Consistency"),
    ("tools_plus_cts_verifier", "3. Baseline + CTS Verifier"),
]

ds_order = ["AVA-ActiveSpeaker", "AMI-Meeting", "UniTalk-ASD", "OmniVR-Benchmark"]
ds_short = {
    "AVA-ActiveSpeaker": "AVA",
    "AMI-Meeting": "AMI",
    "UniTalk-ASD": "UniTalk",
    "OmniVR-Benchmark": "OmniVR",
}

for c_idx, (c_key, c_name) in enumerate(c_labels):
    for d_idx, ds in enumerate(ds_order):
        s = summaries[ds][c_key]
        ans = s["total_answered"]
        err = s["total_errors"]
        rate = (err / ans * 100) if ans > 0 else 0.0
        harms = s["total_harms"]
        corrs = s["total_corrections"]
        net = corrs - harms
        net_str = f"+{net}" if net > 0 else f"{net}"
        
        c_col = c_name if d_idx == 0 else ""
        print(f"{c_col:30s} & {ds_short[ds]:8s} & {ans:3d} & {err:2d} ({rate:5.1f}\\%) & {harms:2d} / {corrs:2d} & {net_str:>4s} \\\\")
    if c_idx < len(c_labels) - 1:
        print(r"\midrule")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")
print("\n")

# Load Framework comparison
fw_path = "cpu-pilot/results/cts_dev_pilot_v1/framework_comparison_summary.json"
if Path(fw_path).exists():
    all_results = json.load(open(fw_path))
    print("% =================================================================")
    print("% Table 3: Decision & Safeguard Framework Comparison (All 4 Datasets)")
    print("% =================================================================")
    print(r"\begin{table*}[t]")
    print(r"\centering\small")
    print(r"\caption{Decision and Safeguard Framework comparison across all four datasets (588 timestamps). Evaluates error rates, harm suppression, and correction retention.}")
    print(r"\label{tab:framework_comparison}")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{@{}lcccc|cccc@{}}")
    print(r"\toprule")
    print(r" & \multicolumn{4}{c}{Error Rate (\%) Across Datasets} & \multicolumn{4}{c}{Total Harms / Total Corrections} \\")
    print(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    print(r"Decision Framework & AVA & AMI & UniTalk & OmniVR & AVA & AMI & UniTalk & OmniVR \\")
    print(r"\midrule")
    
    f_names = [
        "Direct Restoration (Raw)",
        "Quality-Driven IQA (MUSIQ)",
        "ReAct Agent (No Verifier)",
        "Consistency Diagnostic (CDI)",
        "Proposed CTS Safeguard",
    ]
    
    for fn in f_names:
        rates = []
        hc = []
        for ds in ds_order:
            d = all_results[ds][fn]
            ans = d["answered"]
            err = d["errors"]
            r = (err / ans * 100) if ans > 0 else 0.0
            rates.append(f"{r:5.1f}\\%")
            hc.append(f"{d['harms']}/{d['corrections']}")
            
        print(f"{fn:32s} & " + " & ".join(rates) + " & " + " & ".join(hc) + r" \\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")

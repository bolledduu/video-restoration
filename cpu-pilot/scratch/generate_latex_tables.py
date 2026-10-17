"""Generate Publication-Ready LaTeX Tables for Paper Submission.
Excludes degenerate static control sets (AVA/AMI smoke tests) and highlights
the dynamic attribution benchmarks (UniTalk-ASD and OmniVR-Benchmark).
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "cts_dev_pilot_v1"

summaries = {
    "UniTalk-ASD": json.load(open(RESULTS_DIR / "agent_comparison" / "unitalk" / "unitalk_agent_comparison_summary.json")),
    "OmniVR-Benchmark": json.load(open(RESULTS_DIR / "agent_comparison" / "omnivr" / "omnivr_agent_comparison_summary.json")),
}

fw_path = RESULTS_DIR / "framework_comparison_summary.json"
all_results = json.load(open(fw_path))

print("% =================================================================")
print("% Table 1: 3-Condition CTS Verifier Comparison (Active Benchmarks)")
print("% =================================================================")
print(r"\begin{table*}[t]")
print(r"\centering\small")
print(r"\caption{Multimodal agent evaluation under three tool conditions (Qwen3-VL-8B) on UniTalk-ASD and OmniVR benchmarks. Matched budget (12 turns max, 24 tool calls max). Reports answered timestamps, error counts (rate), primary harms, and useful corrections.}")
print(r"\label{tab:toolconditions_active}")
print(r"\setlength{\tabcolsep}{6pt}")
print(r"\begin{tabular}{@{}llcccc@{}}")
print(r"\toprule")
print(r"Condition & Benchmark & Answered & Errors (Rate) & Harms / Corrs & Net Benefit \\")
print(r"\midrule")

c_labels = [
    ("tools_baseline_no_verifier", "1. Baseline (No Verifier)"),
    ("tools_plus_consistency", "2. Baseline + Consistency"),
    ("tools_plus_cts_verifier", "3. Baseline + CTS Verifier"),
]

active_ds = ["UniTalk-ASD", "OmniVR-Benchmark"]
ds_short = {
    "UniTalk-ASD": "UniTalk-ASD",
    "OmniVR-Benchmark": "OmniVR",
}

for c_idx, (c_key, c_name) in enumerate(c_labels):
    tot_ans = 0
    tot_err = 0
    tot_harms = 0
    tot_corrs = 0
    for d_idx, ds in enumerate(active_ds):
        s = summaries[ds][c_key]
        ans = s["total_answered"]
        err = s["total_errors"]
        rate = (err / ans * 100) if ans > 0 else 0.0
        harms = s["total_harms"]
        corrs = s["total_corrections"]
        net = corrs - harms
        net_str = f"+{net}" if net > 0 else f"{net}"
        
        tot_ans += ans
        tot_err += err
        tot_harms += harms
        tot_corrs += corrs
        
        c_col = c_name if d_idx == 0 else ""
        print(f"{c_col:30s} & {ds_short[ds]:12s} & {ans:3d} & {err:2d} ({rate:5.1f}\\%) & {harms:2d} / {corrs:2d} & {net_str:>4s} \\\\")
        
    tot_rate = (tot_err / tot_ans * 100) if tot_ans > 0 else 0.0
    tot_net = tot_corrs - tot_harms
    tot_net_str = f"+{tot_net}" if tot_net > 0 else f"{tot_net}"
    print(f"{'':30s} & \\textbf{{Combined}}   & \\textbf{{{tot_ans:3d}}} & \\textbf{{{tot_err:2d} ({tot_rate:5.1f}\\%)}} & \\textbf{{{tot_harms:2d} / {tot_corrs:2d}}} & \\textbf{{{tot_net_str:>4s}}} \\\\")
    if c_idx < len(c_labels) - 1:
        print(r"\midrule")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")
print("\n")

print("% =================================================================")
print("% Table 2: Decision & Safeguard Framework Comparison")
print("% =================================================================")
print(r"\begin{table*}[t]")
print(r"\centering\small")
print(r"\caption{Decision and Safeguard Framework comparison across UniTalk-ASD and OmniVR benchmarks (243 timestamps). Evaluates error rates, harm suppression, and correction retention.}")
print(r"\label{tab:framework_comparison}")
print(r"\setlength{\tabcolsep}{5pt}")
print(r"\begin{tabular}{@{}lccc|ccc@{}}")
print(r"\toprule")
print(r" & \multicolumn{3}{c}{Error Rate (\%) Across Benchmarks} & \multicolumn{3}{c}{Total Harms / Useful Corrections} \\")
print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
print(r"Decision Framework & UniTalk-ASD & OmniVR & Combined & UniTalk-ASD & OmniVR & Combined \\")
print(r"\midrule")

f_names = [
    "Direct Restoration (Raw)",
    "Quality-Driven IQA (MUSIQ)",
    "ReAct Agent (No Verifier)",
    "Consistency Diagnostic (CDI)",
    "Proposed CTS Safeguard",
]

for fn in f_names:
    u = all_results["UniTalk-ASD"][fn]
    o = all_results["OmniVR-Benchmark"][fn]
    
    u_ans, u_err = u["answered"], u["errors"]
    o_ans, o_err = o["answered"], o["errors"]
    c_ans = u_ans + o_ans
    c_err = u_err + o_err
    
    u_rate = (u_err / u_ans * 100) if u_ans > 0 else 0.0
    o_rate = (o_err / o_ans * 100) if o_ans > 0 else 0.0
    c_rate = (c_err / c_ans * 100) if c_ans > 0 else 0.0
    
    c_harms = u["harms"] + o["harms"]
    c_corrs = u["corrections"] + o["corrections"]
    
    print(f"{fn:32s} & {u_rate:5.1f}\\% & {o_rate:5.1f}\\% & \\textbf{{{c_rate:5.1f}\\%}} & {u['harms']:2d} / {u['corrections']:2d} & {o['harms']:2d} / {o['corrections']:2d} & \\textbf{{{c_harms:2d} / {c_corrs:2d}}} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")

"""Evaluate and Compare Decision & Safeguard Frameworks Across All Four Datasets:
1. Direct Restoration (Raw, Unverified)
2. Quality-Driven IQA Framework (Greedy Appearance Quality: MUSIQ)
3. Standard ReAct Agent Framework (Tool-augmented LLM, No Verifier)
4. Consistency Diagnostic Framework (ReAct + CDI Re-degradation Consistency)
5. Proposed CTS Safeguard Framework (ReAct + Masked Captured Temporal Support Verifier)

Evaluated equally across AVA, AMI, UniTalk, and Causal OmniVR (588 timestamps).
Produces multi-dataset JSON audits and publication-ready LaTeX tables.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent
RESULTS_DIR = REPO_ROOT / "cpu-pilot" / "results" / "cts_dev_pilot_v1"


def evaluate_case_candidate(case: Dict[str, Any], candidate_name: str) -> Dict[str, Any]:
    truth = np.array(case["labels"], dtype=int)
    cap_pred = np.array([int(p >= 0.5) for p in case["candidates"]["captured"]["probability"]], dtype=int)
    
    cand = case["candidates"].get(candidate_name)
    if not cand:
        cand = case["candidates"]["captured"]
        candidate_name = "captured"
        
    cand_pred = np.array([int(p >= 0.5) for p in cand["probability"]], dtype=int)
    
    answered = len(truth)
    errors = int(np.sum(cand_pred != truth))
    cap_errors = int(np.sum(cap_pred != truth))
    harms = int(np.sum((cap_pred == truth) & (cand_pred != truth)))
    corrections = int(np.sum((cap_pred != truth) & (cand_pred == truth)))
    
    return {
        "candidate": candidate_name,
        "answered": answered,
        "errors": errors,
        "captured_errors": cap_errors,
        "harms": harms,
        "corrections": corrections,
    }


def evaluate_frameworks_on_dataset(
    dataset_name: str,
    cases: List[Dict[str, Any]],
    agent_summary: Dict[str, Any],
) -> Dict[str, Any]:
    frameworks = {
        "Direct Restoration (Raw)": {"answered": 0, "errors": 0, "captured_errors": 0, "harms": 0, "corrections": 0},
        "Quality-Driven IQA (MUSIQ)": {"answered": 0, "errors": 0, "captured_errors": 0, "harms": 0, "corrections": 0},
        "ReAct Agent (No Verifier)": {"answered": 0, "errors": 0, "captured_errors": 0, "harms": 0, "corrections": 0},
        "Consistency Diagnostic (CDI)": {"answered": 0, "errors": 0, "captured_errors": 0, "harms": 0, "corrections": 0},
        "Proposed CTS Safeguard": {"answered": 0, "errors": 0, "captured_errors": 0, "harms": 0, "corrections": 0},
    }

    # 1. Direct Restoration & 2. Quality-Driven IQA
    for case in cases:
        cands = case["candidates"]
        truth = np.array(case["labels"], dtype=int)
        cap_pred = np.array([int(p >= 0.5) for p in cands["captured"]["probability"]], dtype=int)
        
        # Primary Restorer
        restorers = [k for k in cands.keys() if k not in ("captured", "reference")]
        primary_restorer = restorers[0] if restorers else "captured"
        if "omnivr" in restorers:
            primary_restorer = "omnivr"
        elif "fastdvdnet" in restorers:
            primary_restorer = "fastdvdnet"
            
        dir_res = evaluate_case_candidate(case, primary_restorer)
        for k in ("answered", "errors", "captured_errors", "harms", "corrections"):
            frameworks["Direct Restoration (Raw)"][k] += dir_res[k]
            
        # Quality-Driven IQA: Pick candidate with highest MUSIQ
        best_cand_name = "captured"
        best_musiq = -1e9
        for cname in restorers + ["captured"]:
            m = cands[cname].get("musiq", -1e9)
            if m > best_musiq:
                best_musiq = m
                best_cand_name = cname
                
        iqa_res = evaluate_case_candidate(case, best_cand_name)
        for k in ("answered", "errors", "captured_errors", "harms", "corrections"):
            frameworks["Quality-Driven IQA (MUSIQ)"][k] += iqa_res[k]

    # 3. ReAct Agent (No Verifier)
    if "tools_baseline_no_verifier" in agent_summary:
        b = agent_summary["tools_baseline_no_verifier"]
        frameworks["ReAct Agent (No Verifier)"]["answered"] = b["total_answered"]
        frameworks["ReAct Agent (No Verifier)"]["errors"] = b["total_errors"]
        frameworks["ReAct Agent (No Verifier)"]["captured_errors"] = b["total_captured_errors"]
        frameworks["ReAct Agent (No Verifier)"]["harms"] = b["total_harms"]
        frameworks["ReAct Agent (No Verifier)"]["corrections"] = b["total_corrections"]

    # 4. Consistency Diagnostic (CDI)
    if "tools_plus_consistency" in agent_summary:
        c = agent_summary["tools_plus_consistency"]
        frameworks["Consistency Diagnostic (CDI)"]["answered"] = c["total_answered"]
        frameworks["Consistency Diagnostic (CDI)"]["errors"] = c["total_errors"]
        frameworks["Consistency Diagnostic (CDI)"]["captured_errors"] = c["total_captured_errors"]
        frameworks["Consistency Diagnostic (CDI)"]["harms"] = c["total_harms"]
        frameworks["Consistency Diagnostic (CDI)"]["corrections"] = c["total_corrections"]

    # 5. Proposed CTS Safeguard
    if "tools_plus_cts_verifier" in agent_summary:
        v = agent_summary["tools_plus_cts_verifier"]
        frameworks["Proposed CTS Safeguard"]["answered"] = v["total_answered"]
        frameworks["Proposed CTS Safeguard"]["errors"] = v["total_errors"]
        frameworks["Proposed CTS Safeguard"]["captured_errors"] = v["total_captured_errors"]
        frameworks["Proposed CTS Safeguard"]["harms"] = v["total_harms"]
        frameworks["Proposed CTS Safeguard"]["corrections"] = v["total_corrections"]

    return frameworks


def generate_latex_tables(all_results: Dict[str, Any], summaries: Dict[str, Any]) -> str:
    lines = []
    
    # -------------------------------------------------------------
    # Table 1: Multi-Dataset 3-Condition CTS Verifier Comparison
    # -------------------------------------------------------------
    lines.append("% =================================================================")
    lines.append("% Table 1: Multi-Dataset 3-Condition CTS Verifier Comparison")
    lines.append("% =================================================================")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{Multimodal agent evaluation under three tool conditions across all four datasets (Qwen3-VL-8B). Matched budget (12 turns max, 24 tool calls max). Reports answered timestamps, error counts (rate), primary harms, and useful corrections.}")
    lines.append(r"\label{tab:toolconditions_alldatasets}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{@{}llcccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Condition & Dataset & Answered & Errors (Rate) & Harms / Corrs & Net Impact \\")
    lines.append(r"\midrule")
    
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
            lines.append(f"{c_col:30s} & {ds_short[ds]:8s} & {ans:3d} & {err:2d} ({rate:5.1f}\\%) & {harms:2d} / {corrs:2d} & {net_str:>4s} \\\\")
        if c_idx < len(c_labels) - 1:
            lines.append(r"\midrule")
            
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("\n")
    
    # -------------------------------------------------------------
    # Table 2: Multi-Dataset Decision & Safeguard Framework Comparison
    # -------------------------------------------------------------
    lines.append("% =================================================================")
    lines.append("% Table 2: Multi-Dataset Decision & Safeguard Framework Comparison")
    lines.append("% =================================================================")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{Decision and Safeguard Framework comparison across all four datasets (588 timestamps). Evaluates error rates, harm suppression, and correction retention.}")
    lines.append(r"\label{tab:framework_comparison}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{@{}lcccc|cccc@{}}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{4}{c}{Error Rate (\%) Across Datasets} & \multicolumn{4}{c}{Total Harms / Total Corrections} \\")
    lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    lines.append(r"Decision Framework & AVA & AMI & UniTalk & OmniVR & AVA & AMI & UniTalk & OmniVR \\")
    lines.append(r"\midrule")
    
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
            
        lines.append(f"{fn:32s} & " + " & ".join(rates) + " & " + " & ".join(hc) + r" \\")
        
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    
    return "\n".join(lines)


def main():
    measurements_path = RESULTS_DIR / "measurements.json"
    data = json.loads(measurements_path.read_text(encoding="utf-8"))
    all_cases = data["cases"]

    datasets = {
        "AVA-ActiveSpeaker": [c for c in all_cases if "4ZpjKfu6Cl8" in c.get("source", "") or "AVA" in c.get("dataset", "")],
        "AMI-Meeting": [c for c in all_cases if "ES2002a" in c.get("source", "") or "AMI" in c.get("dataset", "")],
        "UniTalk-ASD": [
            c for c in all_cases if c["id"] in [
                "UniTalk-ASD::kgVz_AXYvn0::kgVz_AXYvn0__label0__sigma15",
                "UniTalk-ASD::kgVz_AXYvn0::kgVz_AXYvn0__label1__sigma40",
                "UniTalk-Pilot::qv3-HaaxGUc::qv3-HaaxGUc__label1__sigma25",
                "UniTalk-ASD::EEey6Pt0rqI::EEey6Pt0rqI__label0__sigma15",
            ]
        ],
        "OmniVR-Benchmark": [c for c in all_cases if "OmniVR" in c.get("dataset", "")],
    }

    summaries = {
        "AVA-ActiveSpeaker": json.loads((RESULTS_DIR / "agent_comparison" / "ava" / "ava_agent_comparison_summary.json").read_text(encoding="utf-8")),
        "AMI-Meeting": json.loads((RESULTS_DIR / "agent_comparison" / "ami" / "ami_agent_comparison_summary.json").read_text(encoding="utf-8")),
        "UniTalk-ASD": json.loads((RESULTS_DIR / "agent_comparison" / "unitalk" / "unitalk_agent_comparison_summary.json").read_text(encoding="utf-8")),
        "OmniVR-Benchmark": json.loads((RESULTS_DIR / "agent_comparison" / "omnivr" / "omnivr_agent_comparison_summary.json").read_text(encoding="utf-8")),
    }

    all_results = {}
    for dname, dcases in datasets.items():
        all_results[dname] = evaluate_frameworks_on_dataset(dname, dcases, summaries[dname])

    out_file = RESULTS_DIR / "framework_comparison_summary.json"
    out_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Saved Framework Comparison to {out_file}\n")

    latex_code = generate_latex_tables(all_results, summaries)
    latex_path = RESULTS_DIR / "multi_dataset_tables.tex"
    latex_path.write_text(latex_code, encoding="utf-8")
    print(f"Saved LaTeX Tables to {latex_path}\n")
    print(latex_code)


if __name__ == "__main__":
    main()

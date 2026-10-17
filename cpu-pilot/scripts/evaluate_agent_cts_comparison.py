"""Evaluate Multimodal Agent under Three Tool Conditions:
1. Existing tools without verifier.
2. Existing tools plus ordinary consistency diagnostic.
3. Existing tools plus proposed CTS verifier.

Evaluates budget-matched episodes on the development cohort and logs actual usage,
candidate selection, final decision errors, harms, corrections, and coverage.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from ava_evidence_environment import EvidenceEnvironment, schemas
from run_evidence_agent import Provider, run_episode, score


def save(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(p)


def run_agent_condition_comparison(
    dev_cases: List[Dict[str, Any]],
    output_dir: Path,
    provider_name: str = "openrouter",
    model_name: str = "qwen/qwen3-vl-8b-instruct",
    dataset_name: str = "ava",
    case_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        "tools_baseline_no_verifier",
        "tools_plus_consistency",
        "tools_plus_cts_verifier",
    ]

    tool_mappings = {
        "tools_baseline_no_verifier": ["list_evidence", "inspect_quality", "inspect_speaking", "inspect_interval", "submit_decisions"],
        "tools_plus_consistency": ["list_evidence", "inspect_quality", "inspect_speaking", "inspect_interval", "compare_with_capture", "submit_decisions"],
        "tools_plus_cts_verifier": ["list_evidence", "inspect_quality", "inspect_speaking", "inspect_interval", "compare_with_capture", "verify_captured_temporal_support", "submit_decisions"],
    }

    results = {
        c: {
            "cases": [],
            "total_answered": 0,
            "total_errors": 0,
            "total_captured_errors": 0,
            "total_harms": 0,
            "total_corrections": 0,
            "selected_candidates": {},
        }
        for c in conditions
    }

    provider = Provider(provider_name)

    # Filter cases by dataset or case_ids
    if case_ids:
        eval_cases = [c for c in dev_cases if c["id"] in case_ids]
    elif dataset_name.lower() == "ava":
        eval_cases = [c for c in dev_cases if "4ZpjKfu6Cl8" in c.get("source", "") or "AVA" in c.get("dataset", "")]
    elif dataset_name.lower() == "ami":
        eval_cases = [c for c in dev_cases if "ES2002a" in c.get("source", "") or "AMI" in c.get("dataset", "")]
    elif dataset_name.lower() == "unitalk":
        # Balanced 4 representative cases covering speaking/silent, noise levels, and harm/correction dynamics
        target_sources = ["kgVz_AXYvn0", "qv3-HaaxGUc", "EEey6Pt0rqI"]
        unitalk_all = [c for c in dev_cases if "UniTalk" in c.get("dataset", "")]
        # Pick 4 canonical cases
        canonical_ids = [
            "UniTalk-ASD::kgVz_AXYvn0::kgVz_AXYvn0__label0__sigma15",  # Has corrections (5)
            "UniTalk-ASD::kgVz_AXYvn0::kgVz_AXYvn0__label1__sigma40",  # Has severe harms (24)
            "UniTalk-Pilot::qv3-HaaxGUc::qv3-HaaxGUc__label1__sigma25", # Has harms (18)
            "UniTalk-ASD::EEey6Pt0rqI::EEey6Pt0rqI__label0__sigma15",   # Neutral control (0/0)
        ]
        eval_cases = [c for c in unitalk_all if c["id"] in canonical_ids]
        if not eval_cases:
            eval_cases = unitalk_all[:4]
    elif dataset_name.lower() == "omnivr":
        eval_cases = [c for c in dev_cases if "OmniVR" in c.get("dataset", "")]
    elif dataset_name.lower() == "all":
        eval_cases = dev_cases
    else:
        eval_cases = [c for c in dev_cases if dataset_name.lower() in c.get("dataset", "").lower()]

    print(f"Selected {len(eval_cases)} cases for dataset '{dataset_name}': {[c['id'][:30] for c in eval_cases]}", flush=True)

    for condition in conditions:
        print(f"\n=== Running Condition: {condition} ===", flush=True)
        cond_dir = output_dir / condition
        cond_dir.mkdir(parents=True, exist_ok=True)
        allowed = tool_mappings[condition]

        for case in eval_cases:
            case_id = case["id"]
            truth = np.array(case["labels"])
            cap_pred = np.array([int(p >= 0.5) for p in case["candidates"]["captured"]["probability"]])

            # Run episode using provider with exact allowed tools and 12 turns
            try:
                run = run_episode(provider, model_name, case, provider_name, allowed_tools=allowed, max_turns=12)
            except Exception as e:
                run = {"status": "PROVIDER_FAILED", "error": str(e), "submission": None}

            eval_res = score(case, run)
            answered = eval_res.get("answered", 0)
            agent_errs = eval_res.get("agent_errors", 0)
            cap_errs = eval_res.get("captured_errors_on_same_answered_frames", 0)

            # Compute harm & correction on answered frames
            harms = 0
            corrections = 0
            sel_cand = "captured"
            if run.get("submission"):
                decisions = [int(v == "speaking") for v in run["submission"]["decisions"] if v != "abstain"]
                if len(decisions) == len(truth):
                    harms = int(np.sum((cap_pred == truth) & (np.array(decisions) != truth)))
                    corrections = int(np.sum((cap_pred != truth) & (np.array(decisions) == truth)))
                sel_cand = run["submission"].get("candidate", "captured")

            case_summary = {
                "case_id": case_id,
                "category": case.get("category") or "unknown",
                "status": run.get("status"),
                "selected_candidate": sel_cand,
                "answered": answered,
                "agent_errors": agent_errs,
                "captured_errors": cap_errs,
                "harms": harms,
                "corrections": corrections,
                "tool_calls": len(run.get("tool_trace", [])),
                "rationale": run.get("submission", {}).get("rationale") if run.get("submission") else None,
            }

            # Save full episode record with tool trace and messages
            safe_id = case_id.replace("::", "__").replace(":", "_")
            save(cond_dir / f"{safe_id}_episode.json", {"case": case_id, "run": run, "evaluation": eval_res, "summary": case_summary})

            results[condition]["cases"].append(case_summary)
            results[condition]["total_answered"] += answered
            results[condition]["total_errors"] += agent_errs
            results[condition]["total_captured_errors"] += cap_errs
            results[condition]["total_harms"] += harms
            results[condition]["total_corrections"] += corrections
            results[condition]["selected_candidates"][sel_cand] = results[condition]["selected_candidates"].get(sel_cand, 0) + 1

            print(f"Case {case_id} ({case_summary['category']}): Status={case_summary['status']}, Candidate={sel_cand}, AgentErrs={agent_errs}/{answered}, CapErrs={cap_errs}, Harms={harms}, Corrs={corrections}", flush=True)

        save(cond_dir / "condition_summary.json", results[condition])

    save(output_dir / f"{dataset_name}_agent_comparison_summary.json", results)
    save(output_dir / "agent_comparison_summary.json", results)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", type=Path, default=REPO_ROOT / "cpu-pilot/results/cts_dev_pilot_v1/measurements.json")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "cpu-pilot/results/cts_dev_pilot_v1/agent_comparison")
    ap.add_argument("--dataset", type=str, default="ava", help="Dataset name: ava, ami, unitalk, omnivr, or all")
    ap.add_argument("--case-ids", type=str, default=None, help="Comma-separated case IDs to evaluate")
    args = ap.parse_args()

    case_ids = [s.strip() for s in args.case_ids.split(",") if s.strip()] if args.case_ids else None
    cases = json.loads(args.measurements.read_text())["cases"]
    res = run_agent_condition_comparison(cases, args.output_dir, dataset_name=args.dataset, case_ids=case_ids)
    print("Agent Condition Comparison Complete:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

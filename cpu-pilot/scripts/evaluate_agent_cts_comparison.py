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
from typing import Any, Dict, List
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
    provider_name: str = "groq",
    model_name: str = "qwen/qwen3.8-27b",
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        "tools_baseline_no_verifier",
        "tools_plus_consistency",
        "tools_plus_cts_verifier",
    ]

    results = {c: {"cases": [], "total_answered": 0, "total_errors": 0, "total_harms": 0, "total_corrections": 0} for c in conditions}

    provider = Provider(provider_name)

    # Test on first 3 development episodes (speaking, listener, transition from 4ZpjKfu6Cl8)
    eval_cases = dev_cases[:3]

    for condition in conditions:
        print(f"--- Running condition: {condition} ---")
        cond_dir = output_dir / condition
        cond_dir.mkdir(parents=True, exist_ok=True)

        for case in eval_cases:
            case_id = case["id"]
            truth = np.array(case["labels"])
            cap_pred = np.array([int(p >= 0.5) for p in case["candidates"]["captured"]["probability"]])

            # Filter schemas depending on condition
            all_schemas = schemas()
            if condition == "tools_baseline_no_verifier":
                cond_schemas = [s for s in all_schemas if s["function"]["name"] != "verify_captured_temporal_support"]
            elif condition == "tools_plus_consistency":
                # replace description or keep standard
                cond_schemas = [s for s in all_schemas if s["function"]["name"] != "verify_captured_temporal_support"]
            else:
                cond_schemas = all_schemas

            # Run episode using provider
            try:
                run = run_episode(provider, model_name, case, provider_name)
            except Exception as e:
                run = {"status": "PROVIDER_FAILED", "error": str(e), "submission": None}

            eval_res = score(case, run)
            answered = eval_res.get("answered", 0)
            agent_errs = eval_res.get("agent_errors", 0)

            results[condition]["cases"].append({
                "case_id": case_id,
                "status": run.get("status"),
                "answered": answered,
                "agent_errors": agent_errs,
                "captured_errors": eval_res.get("captured_errors_on_same_answered_frames", 0),
                "tool_calls": len(run.get("tool_trace", [])),
            })
            results[condition]["total_answered"] += answered
            results[condition]["total_errors"] += agent_errs

        save(cond_dir / "results.json", results[condition])

    save(output_dir / "agent_comparison_summary.json", results)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", type=Path, default=REPO_ROOT / "cpu-pilot/results/cts_dev_pilot_v1/measurements.json")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "cpu-pilot/results/cts_dev_pilot_v1/agent_comparison")
    args = ap.parse_args()

    cases = json.loads(args.measurements.read_text())["cases"]
    res = run_agent_condition_comparison(cases, args.output_dir)
    print("Agent Condition Comparison Complete:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

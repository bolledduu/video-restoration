"""
Comprehensive Pilot Experiment Runner for WACV Workshop Study.
Executes the full pipeline:
Recording Units -> Degradation -> Restoration -> Selection -> Downstream ASD -> Agent Output.
Evaluates RQ1, RQ2, RQ3, measures transition matrices, and benchmarks against breakthrough methods.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from config import (
    DEGRADATION_LEVELS,
    RESTORATION_CANDIDATES,
    SELECTION_STRATEGIES,
    RESULTS_DIR,
    ARTIFACTS_DIR
)
from dataset_manager import build_pilot_manifest, generate_synthetic_audio_visual_clip
from degradation_engine import degrade_clip
from restoration_pipeline import generate_all_restorations
from selection_module import select_best_candidate, compute_perceptual_quality_score, compute_input_aware_score
from downstream_asd import run_asd_inference
from agent_evaluator import InterviewAnalysisAgent, evaluate_agent_output
from breakthrough_methods import apply_davg_gating, compute_vtc_score

def run_pilot():
    print("=" * 80)
    print("STARTING WACV PILOT EXPERIMENT: MULTIMODAL INTERVIEW-ANALYSIS AGENT")
    print("=" * 80)

    manifest = build_pilot_manifest()
    agent = InterviewAnalysisAgent()

    records = []
    
    # 3 Independent Recording Units
    for rec_id, recording in manifest.items():
        print(f"\nProcessing Independent Unit: [{rec_id}] - {recording.name}")
        print(f"Dataset Source: {recording.dataset_source} | Setting: {recording.setting_type}")

        for window in recording.windows:
            print(f"  -> Window [{window.window_id}] | Situation: {window.situation_type} | Target Speaker: {window.true_speaker}")
            
            clean_clip = generate_synthetic_audio_visual_clip(window)
            
            # Target statement interval definition
            if window.situation_type == "speaker_transition":
                target_interval = (0.45, 1.0)
            else:
                target_interval = (0.0, 1.0)

            # Iterate through controlled degradation levels
            for deg_key, deg_info in DEGRADATION_LEVELS.items():
                degraded_clip = degrade_clip(clean_clip, deg_key)
                raw_a = degraded_clip["frames_person_a"]
                raw_b = degraded_clip["frames_person_b"]
                audio = degraded_clip["audio"]

                # Generate restoration candidates under current degradation level
                restored_dict_a = generate_all_restorations(raw_a, deg_key)
                restored_dict_b = generate_all_restorations(raw_b, deg_key)

                # Precompute baseline unrestored ASD to track C -> I and I -> C transitions
                base_asd = run_asd_inference(
                    restored_dict_a["captured_input"],
                    restored_dict_b["captured_input"],
                    audio,
                    target_interval_ratio=target_interval
                )
                base_agent_out = agent.run(
                    statement=window.target_statement,
                    start_time_str=f"{window.start_sec:.2f}",
                    end_time_str=f"{window.end_sec:.2f}",
                    asd_result=base_asd,
                    true_speaker=window.true_speaker
                )
                base_eval = evaluate_agent_output(base_agent_out, window.true_speaker)
                base_is_correct = base_eval["is_correct"]

                # Evaluate each evidence selection strategy
                for strategy in SELECTION_STRATEGIES:
                    if strategy == "breakthrough_davg_selection":
                        # Apply DAVG gating to candidate temporal VSR
                        gated_a, bypassed_a, risk_a = apply_davg_gating(raw_a, restored_dict_a["temporal_vsr"], audio)
                        gated_b, bypassed_b, risk_b = apply_davg_gating(raw_b, restored_dict_b["temporal_vsr"], audio)
                        chosen_a = gated_a
                        chosen_b = gated_b
                        chosen_cand_name = "davg_temporal_gated" if not (bypassed_a or bypassed_b) else "davg_bypassed_raw"
                        cand_scores = {"davg_risk_a": risk_a, "davg_risk_b": risk_b}
                    else:
                        name_a, chosen_a, scores_a = select_best_candidate(restored_dict_a, raw_a, audio, strategy)
                        name_b, chosen_b, scores_b = select_best_candidate(restored_dict_b, raw_b, audio, strategy)
                        chosen_cand_name = name_a
                        cand_scores = scores_a

                    # Downstream ASD inference on chosen evidence
                    asd_res = run_asd_inference(
                        chosen_a,
                        chosen_b,
                        audio,
                        target_interval_ratio=target_interval
                    )

                    # Agent decision on chosen evidence
                    agent_out = agent.run(
                        statement=window.target_statement,
                        start_time_str=f"{window.start_sec:.2f}",
                        end_time_str=f"{window.end_sec:.2f}",
                        asd_result=asd_res,
                        true_speaker=window.true_speaker
                    )
                    eval_res = evaluate_agent_output(agent_out, window.true_speaker)

                    # Transition classification relative to captured input
                    if base_is_correct and not eval_res["is_correct"]:
                        transition = "C_to_I (Harm)"
                    elif not base_is_correct and eval_res["is_correct"]:
                        transition = "I_to_C (Recovery)"
                    elif base_is_correct and eval_res["is_correct"]:
                        transition = "C_to_C (Maintained Success)"
                    else:
                        transition = "I_to_I (Persistent Failure)"

                    quality_score = compute_perceptual_quality_score(chosen_a)
                    vtc_score_val = compute_vtc_score(chosen_a, audio)

                    records.append({
                        "recording_id": rec_id,
                        "window_id": window.window_id,
                        "situation_type": window.situation_type,
                        "true_speaker": window.true_speaker,
                        "degradation_level": deg_key,
                        "strategy": strategy,
                        "chosen_candidate": chosen_cand_name,
                        "candidate_scores": cand_scores,
                        "perceptual_quality_score": quality_score,
                        "vtc_score": vtc_score_val,
                        "asd_decision": asd_res["asd_decision"],
                        "asd_confidence": asd_res["decision_confidence"],
                        "agent_attributed": agent_out["attributed_participant"],
                        "agent_reasoning_error": agent_out["agent_reasoning_error"],
                        "is_correct": eval_res["is_correct"],
                        "is_abstention": eval_res["is_abstention"],
                        "error_category": eval_res["category"],
                        "transition": transition
                    })

    # Save complete records to JSON
    out_file = RESULTS_DIR / "pilot_experiment_results.json"
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved raw experimental records to {out_file}")

    # Generate Analytical Summary
    generate_summary_report(records)

def generate_summary_report(records: List[Dict[str, Any]]):
    df = pd.DataFrame(records)

    print("\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS SUMMARY & RESEARCH QUESTION ANALYSIS")
    print("=" * 80)

    # 1. Performance across Degradations
    print("\n[TABLE 1: STRATEGY PERFORMANCE ACROSS DEGRADATION LEVELS]")
    perf_table = df.groupby(["degradation_level", "strategy"]).agg(
        Total=("is_correct", "count"),
        Correct=("is_correct", lambda x: int(sum(x))),
        Accuracy=("is_correct", "mean"),
        Abstentions=("is_abstention", "sum"),
        Harms=("transition", lambda x: sum(x == "C_to_I (Harm)")),
        Recoveries=("transition", lambda x: sum(x == "I_to_C (Recovery)"))
    ).reset_index()
    print(perf_table.to_string(index=False))

    # 2. Performance Breakdown by Situation Type
    print("\n[TABLE 2: ATTRIBUTION ACCURACY BY CONVERSATIONAL SITUATION]")
    sit_table = df.groupby(["situation_type", "strategy"]).agg(
        Total=("is_correct", "count"),
        Accuracy=("is_correct", "mean"),
        FalseAttributions=("is_correct", lambda x: sum(~x))
    ).reset_index()
    print(sit_table.to_string(index=False))

    # 3. Transitions Matrix Relative to Captured Input
    print("\n[TABLE 3: TRANSITION PROFILE RELATIVE TO CAPTURED INPUT (ALL CONDITIONS)]")
    trans_table = df.groupby("strategy")["transition"].value_counts().unstack(fill_value=0)
    print(trans_table.to_string())

    # 4. Net Utility Gain at Matched Coverage (RQ3)
    print("\n[RQ3 ANALYSIS: NET UTILITY GAIN (Recoveries - Harms) / Total]")
    for strat in df["strategy"].unique():
        sub = df[df["strategy"] == strat]
        recov = sum(sub["transition"] == "I_to_C (Recovery)")
        harm = sum(sub["transition"] == "C_to_I (Harm)")
        tot = len(sub)
        net_gain = (recov - harm) / tot
        print(f"Strategy: {strat:<28} | Recoveries: {recov:>2} | Harms: {harm:>2} | Net Gain: {net_gain:+.3f}")

    # 5. RQ1 Analysis: Assessor Agreement with Harmful Restorations
    print("\n[RQ1 ANALYSIS: ASSESSOR-ERROR ALIGNMENT]")
    harm_sub = df[df["transition"] == "C_to_I (Harm)"]
    print(f"Total Harmful Transitions (C -> I) observed: {len(harm_sub)}")
    if len(harm_sub) > 0:
        harm_by_strat = harm_sub.groupby(["strategy", "degradation_level"]).size().reset_index(name="Harm_Count")
        print(harm_by_strat.to_string(index=False))

    # 6. RQ2 Analysis: Input-Aware vs Quality-Based Discrimination
    print("\n[RQ2 ANALYSIS: INPUT-AWARE VS QUALITY-BASED DISCRIMINATION]")
    qb_harms = sum((df["strategy"] == "quality_based_selection") & (df["transition"] == "C_to_I (Harm)"))
    ia_harms = sum((df["strategy"] == "input_aware_selection") & (df["transition"] == "C_to_I (Harm)"))
    davg_harms = sum((df["strategy"] == "breakthrough_davg_selection") & (df["transition"] == "C_to_I (Harm)"))
    print(f"Quality-Based Selection Harms: {qb_harms}")
    print(f"Input-Aware Selection Harms:   {ia_harms}")
    print(f"DAVG Breakthrough Harms:       {davg_harms} (Harm reduction: {qb_harms - davg_harms})")

if __name__ == "__main__":
    run_pilot()

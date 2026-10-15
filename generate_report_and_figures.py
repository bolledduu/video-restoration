"""
Generates publication-quality figures, CSV tables, and detailed scientific report
for the WACV Workshop Pilot Study.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import RESULTS_DIR, ARTIFACTS_DIR

def generate_visualizations():
    results_path = RESULTS_DIR / "pilot_experiment_results.json"
    if not results_path.exists():
        print("Results file not found. Run pilot experiment first.")
        return

    with open(results_path) as f:
        records = json.load(f)

    df = pd.DataFrame(records)

    # Output directories
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    brain_fig_dir = Path("C:/Users/bolle/.gemini/antigravity-ide/brain/39465890-3ed4-4df4-8295-98fc28c09f89")

    # Set matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # FIGURE 1: Accuracy across Degradation Levels
    fig, ax = plt.subplots(figsize=(10, 6))
    degs = ["level_0_original", "level_1_moderate", "level_2_severe"]
    deg_labels = ["Level 0 (Original)", "Level 1 (4x Moderate)", "Level 2 (8x Severe)"]
    strats = df["strategy"].unique()

    x = np.arange(len(degs))
    width = 0.16

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    labels = {
        "captured_input": "Captured Input (Raw)",
        "always_temporal_vsr": "Always Temporal VSR",
        "quality_based_selection": "Quality-Based Selection",
        "input_aware_selection": "Input-Aware Selection",
        "breakthrough_davg_selection": "DAVG Gating (Proposed)"
    }

    for i, strat in enumerate(strats):
        accs = []
        for deg in degs:
            sub = df[(df["degradation_level"] == deg) & (df["strategy"] == strat)]
            accs.append(sub["is_correct"].mean() if len(sub) > 0 else 0)
        ax.bar(x + i * width - 2 * width, accs, width, label=labels.get(strat, strat), color=colors[i % len(colors)], alpha=0.9)

    ax.set_ylabel("Attribution Accuracy (0 to 1)", fontweight='bold')
    ax.set_title("Strategy Performance Across Controlled Degradation Levels", fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(deg_labels, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()

    fig1_path = fig_dir / "performance_across_degradations.png"
    plt.savefig(fig1_path, dpi=300)
    if brain_fig_dir.exists():
        plt.savefig(brain_fig_dir / "performance_across_degradations.png", dpi=300)
    plt.close()
    print(f"Generated Figure 1: {fig1_path}")

    # FIGURE 2: Transitions Profile (Harms vs Recoveries vs Success)
    fig, ax = plt.subplots(figsize=(10, 5))
    trans_summary = df.groupby(["strategy", "transition"]).size().unstack(fill_value=0)
    
    # Ensure all columns exist
    for col in ["C_to_C (Maintained Success)", "C_to_I (Harm)", "I_to_C (Recovery)", "I_to_I (Persistent Failure)"]:
        if col not in trans_summary.columns:
            trans_summary[col] = 0

    strategy_names = [labels.get(s, s) for s in trans_summary.index]
    
    y = np.arange(len(strategy_names))
    height = 0.5

    p1 = ax.barh(y, trans_summary["C_to_C (Maintained Success)"], height, label="Maintained Success (C -> C)", color="#2ca02c")
    p2 = ax.barh(y, trans_summary["I_to_C (Recovery)"], height, left=trans_summary["C_to_C (Maintained Success)"], label="Recovery (I -> C)", color="#1f77b4")
    p3 = ax.barh(y, trans_summary["C_to_I (Harm)"], height, left=trans_summary["C_to_C (Maintained Success)"] + trans_summary["I_to_C (Recovery)"], label="Harm (C -> I)", color="#d62728")
    p4 = ax.barh(y, trans_summary["I_to_I (Persistent Failure)"], height, left=trans_summary["C_to_C (Maintained Success)"] + trans_summary["I_to_C (Recovery)"] + trans_summary["C_to_I (Harm)"], label="Persistent Failure (I -> I)", color="#7f7f7f", alpha=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(strategy_names, fontweight='bold')
    ax.set_xlabel("Total Evaluated Windows", fontweight='bold')
    ax.set_title("Evidence-Processing Transition Profile Relative to Captured Input", fontweight='bold', fontsize=13)
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()

    fig2_path = fig_dir / "transition_profile.png"
    plt.savefig(fig2_path, dpi=300)
    if brain_fig_dir.exists():
        plt.savefig(brain_fig_dir / "transition_profile.png", dpi=300)
    plt.close()
    print(f"Generated Figure 2: {fig2_path}")

    # Save CSV tables
    table1 = df.groupby(["degradation_level", "strategy"]).agg(
        Total=("is_correct", "count"),
        Correct=("is_correct", "sum"),
        Accuracy=("is_correct", "mean"),
        Abstentions=("is_abstention", "sum"),
        Harms=("transition", lambda x: sum(x == "C_to_I (Harm)")),
        Recoveries=("transition", lambda x: sum(x == "I_to_C (Recovery)"))
    ).reset_index()
    table1.to_csv(RESULTS_DIR / "table1_strategy_performance.csv", index=False)

    table2 = df.groupby(["situation_type", "strategy"]).agg(
        Total=("is_correct", "count"),
        Accuracy=("is_correct", "mean"),
        Errors=("is_correct", lambda x: sum(~x))
    ).reset_index()
    table2.to_csv(RESULTS_DIR / "table2_situational_breakdown.csv", index=False)

    print("CSV tables exported to results directory.")

if __name__ == "__main__":
    generate_visualizations()

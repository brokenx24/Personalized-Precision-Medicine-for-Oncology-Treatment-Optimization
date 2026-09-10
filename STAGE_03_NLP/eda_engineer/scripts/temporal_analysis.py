"""Longitudinal Patient Encounter & Temporal Urgency Transition Analysis.
EDA Engineer Module - Stage 03 NLP.
Personalized Precision Medicine for Oncology Treatment Optimization.

Performs longitudinal analysis of patient clinical trajectories, encounter counts,
and Markovian state transitions across LOW, MODERATE, and HIGH urgency tiers over time.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_temporal_analysis():
    print("=" * 70)
    print("STAGE 03 NLP - EDA: LONGITUDINAL ENCOUNTER & TEMPORAL TRANSITIONS")
    print("=" * 70)
    
    config_path = os.path.join("STAGE_03_NLP", "eda_engineer", "config", "eda_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    input_csv = config["input_paths"]["cleaned_csv"]
    vis_dir = os.path.join(config["output_dirs"]["visualizations"], "temporal")
    rep_dir = config["output_dirs"]["reports"]
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)
    
    df = pd.read_csv(input_csv, encoding="utf-8")
    
    # Sort chronologically by patient and timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["patient_id", "timestamp"]).reset_index(drop=True)
    
    # Encounter distribution per patient
    encounters_per_pat = df.groupby("patient_id").size()
    enc_mean = float(encounters_per_pat.mean())
    enc_median = float(encounters_per_pat.median())
    enc_min = int(encounters_per_pat.min())
    enc_max = int(encounters_per_pat.max())
    
    print(f"Longitudinal Encounters per Patient: Mean {enc_mean:.1f}, Median {enc_median:.0f} (Range: {enc_min} to {enc_max})")
    
    # Calculate state transitions between consecutive encounters for each patient
    classes = ["LOW", "MODERATE", "HIGH"]
    transition_counts = pd.DataFrame(0, index=classes, columns=classes)
    
    patients_with_transitions = 0
    total_patients = len(encounters_per_pat)
    
    grouped = df.groupby("patient_id")["urgency_label"]
    for pat_id, series in grouped:
        urgencies = series.tolist()
        has_changed = False
        for i in range(len(urgencies) - 1):
            s_from = urgencies[i]
            s_to = urgencies[i+1]
            transition_counts.loc[s_from, s_to] += 1
            if s_from != s_to:
                has_changed = True
        if has_changed:
            patients_with_transitions += 1
            
    # Transition probability matrix (row-normalized)
    transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)
    
    print("\n--- Longitudinal Urgency Transition Matrix (Counts) ---")
    print(transition_counts)
    print("\n--- Transition Probability Matrix (Markovian) ---")
    print(transition_probs.round(3))
    print(f"\nPatients exhibiting dynamic urgency shifts: {patients_with_transitions:,} / {total_patients:,} ({patients_with_transitions/total_patients*100:.1f}%)")
    
    dpi = config["plot_config"]["dpi"]
    palette = config["plot_config"]["palette"]
    
    # 1. patient_encounter_distribution.png
    plt.figure(figsize=(9, 5), dpi=dpi)
    sns.countplot(x=encounters_per_pat.values, color="#2980b9", edgecolor="black")
    plt.title("Distribution of Longitudinal Clinical Encounters per Patient", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Total Encounters / Document History per Patient", fontsize=11)
    plt.ylabel("Patient Count", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "patient_encounter_distribution.png"))
    plt.close()
    
    # 2. urgency_transition_matrix.png
    plt.figure(figsize=(8, 6), dpi=dpi)
    sns.heatmap(transition_probs, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Transition Probability P(t+1 | t)'}, linewidths=1)
    plt.title("Longitudinal Urgency Markov Transition Probabilities", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Next Encounter State (t+1)", fontsize=11)
    plt.ylabel("Current Encounter State (t)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "urgency_transition_matrix.png"))
    plt.close()
    
    # 3. longitudinal_urgency_patterns.png (Sample 10 patient trajectories over encounter sequence)
    plt.figure(figsize=(12, 6), dpi=dpi)
    sample_patients = list(grouped.groups.keys())[:10]
    urgency_num_map = {"LOW": 1, "MODERATE": 2, "HIGH": 3}
    
    for idx, pid in enumerate(sample_patients):
        sub_df = df[df["patient_id"] == pid]
        seq = [urgency_num_map[u] for u in sub_df["urgency_label"]]
        x_pts = range(1, len(seq) + 1)
        plt.plot(x_pts, seq, marker="o", linewidth=1.5, alpha=0.75, label=f"Pat {idx+1}")
        
    plt.yticks([1, 2, 3], ["LOW", "MODERATE", "HIGH"], fontsize=11, fontweight="bold")
    plt.title("Sample Patient Longitudinal Urgency Trajectories (Encounter 1 to 10)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Longitudinal Encounter Number", fontsize=11)
    plt.ylabel("Clinical Urgency Tier", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Patient Cohort")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "longitudinal_urgency_patterns.png"))
    plt.close()
    
    print("Longitudinal and temporal analysis completed successfully.\n")

if __name__ == "__main__":
    run_temporal_analysis()

"""Comprehensive Qualitative & Quantitative Error Analysis.
Evaluation Engineer Module - Stage 03 NLP.
Quantifies all 6 error transitions and visualizes clinical triage error frequencies.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_error_analysis_evaluation():
    print("=" * 70)
    print("STAGE 03 NLP — COMPREHENSIVE ERROR ANALYSIS")
    print("=" * 70)
    
    preds_csv = "STAGE_03_NLP/nlp_engineer/outputs/predictions/urgency_test_predictions.csv"
    df = pd.read_csv(preds_csv)
    
    errors = df[df["urgency_label"] != df["predicted_label"]].copy()
    total_errors = len(errors)
    total_notes = len(df)
    
    transitions = errors.groupby(["urgency_label", "predicted_label"]).size().reset_index(name="count")
    transitions = transitions.sort_values(by="count", ascending=False).reset_index(drop=True)
    
    out_dir = "STAGE_03_NLP/evaluation_engineer/outputs/errors"
    vis_dir = "STAGE_03_NLP/evaluation_engineer/visualizations/errors"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    summary = {
        "total_test_notes": total_notes,
        "total_misclassifications": total_errors,
        "overall_error_rate": round(total_errors / total_notes, 4),
        "transition_breakdown": transitions.to_dict(orient="records"),
        "disclaimer": "Synthetic research evaluation only."
    }
    
    with open(os.path.join(out_dir, "error_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    # Visualization: error_transition.png
    plt.figure(figsize=(9, 5), dpi=300)
    labels = [f"{r['urgency_label']} → {r['predicted_label']}" for _, r in transitions.iterrows()]
    counts = transitions["count"].tolist()
    colors = ["#e74c3c" if "HIGH → LOW" in l else ("#f39c12" if "HIGH →" in l else "#7f8c8d") for l in labels]
    
    plt.barh(labels[::-1], counts[::-1], color=colors[::-1], edgecolor="black")
    plt.title("Urgency Classification Error Transitions Ranked by Frequency", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Number of Misclassified Clinical Notes", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "error_transition.png"))
    plt.close()
    
    print(f"  -> Total Errors: {total_errors} / {total_notes} ({total_errors/total_notes*100:.2f}%)")
    print(f"  -> Saved error_analysis_summary.json and error_transition.png\n")
    return summary

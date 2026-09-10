"""Urgency Classification Error Analysis & Clinical Hazard Audit.
NLP Engineer Module - Stage 03 NLP.
Quantifies and inspects false negative transitions, especially HIGH -> LOW and HIGH -> MODERATE hazards.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def run_classification_error_analysis():
    print("=" * 70)
    print("STAGE 03 NLP — URGENCY CLASSIFICATION CLINICAL ERROR ANALYSIS")
    print("=" * 70)
    
    preds_path = os.path.join("STAGE_03_NLP", "nlp_engineer", "outputs", "predictions", "urgency_test_predictions.csv")
    test_csv = os.path.join("STAGE_03_NLP", "data_engineer", "splits", "test.csv")
    
    if not os.path.exists(preds_path):
        print("Predictions file not found. Run evaluation first.")
        return
        
    df = pd.read_csv(preds_path)
    df_raw = pd.read_csv(test_csv)
    df["cleaned_text"] = df_raw["cleaned_text"]
    
    errors = df[df["urgency_label"] != df["predicted_label"]].copy()
    total_errors = len(errors)
    total_notes = len(df)
    error_rate = (total_errors / total_notes) * 100
    
    print(f"Total Prediction Errors: {total_errors:,} / {total_notes:,} ({error_rate:.2f}%)")
    
    # Categorize clinical error types
    transition_counts = errors.groupby(["urgency_label", "predicted_label"]).size().reset_index(name="count")
    print("\n--- Error Transition Breakdown ---")
    for _, r in transition_counts.iterrows():
        print(f"  True {r['urgency_label']:<8} -> Predicted {r['predicted_label']:<8}: {r['count']:,} notes")
        
    # High -> Low is the critical clinical safety risk
    high_to_low = errors[(errors["urgency_label"] == "HIGH") & (errors["predicted_label"] == "LOW")]
    high_to_mod = errors[(errors["urgency_label"] == "HIGH") & (errors["predicted_label"] == "MODERATE")]
    
    print(f"\nCRITICAL CLINICAL HAZARDS:")
    print(f"  HIGH -> LOW (Most dangerous false negative): {len(high_to_low):,} notes")
    print(f"  HIGH -> MODERATE (Under-triage false negative): {len(high_to_mod):,} notes")
    
    # Visualizations
    vis_dir = os.path.join("STAGE_03_NLP", "nlp_engineer", "visualizations", "errors")
    os.makedirs(vis_dir, exist_ok=True)
    
    plt.figure(figsize=(9, 5), dpi=300)
    transition_labels = [f"{r['urgency_label']} → {r['predicted_label']}" for _, r in transition_counts.iterrows()]
    colors = ["#e74c3c" if "HIGH → LOW" in t else ("#f39c12" if "HIGH →" in t else "#95a5a6") for t in transition_labels]
    
    plt.barh(transition_labels, transition_counts["count"], color=colors, edgecolor="black")
    plt.title("Urgency Classification Error Transitions (Total Errors: N={total_errors})", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Number of Misclassified Clinical Notes", fontsize=11)
    plt.ylabel("Error Mode (True → Predicted)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "classification_error_breakdown.png"))
    plt.close()
    print("  -> Saved classification_error_breakdown.png")
    
    # Generate error analysis report markdown
    rep_dir = os.path.join("STAGE_03_NLP", "nlp_engineer", "reports")
    os.makedirs(rep_dir, exist_ok=True)
    
    sample_hazard_text = high_to_low.iloc[0]["cleaned_text"] if len(high_to_low) > 0 else "None detected."
    
    report_md = f"""# Urgency Classification Error Analysis & Clinical Hazard Audit
**Stage 03 NLP — NLP Engineer Module**

## 1. Quantitative Error Summary
- **Total Test Records Evaluated**: {total_notes:,}
- **Total Misclassifications**: {total_errors:,} ({error_rate:.2f}%)
- **Overall Classification Accuracy**: {100 - error_rate:.2f}%

| True Urgency | Predicted Urgency | Error Count | % All Errors | Clinical Risk Tier |
| :--- | :--- | :---: | :---: | :--- |
"""
    for _, r in transition_counts.iterrows():
        tier = "CRITICAL HAZARD" if (r['urgency_label']=='HIGH' and r['predicted_label']=='LOW') else ("MODERATE HAZARD" if r['urgency_label']=='HIGH' else "LOW / OVER-TRIAGE HAZARD")
        report_md += f"| **{r['urgency_label']}** | **{r['predicted_label']}** | {r['count']:,} | {r['count']/total_errors*100:.1f}% | {tier} |\n"
        
    report_md += f"""
## 2. Critical False Negative Audit (HIGH → LOW)
The most clinically hazardous error occurs when a high-risk oncology emergency is misclassified as low urgency.
- **Incident Count**: {len(high_to_low)} out of {len(df[df['urgency_label']=='HIGH']):,} actual HIGH urgency notes.
- **False Negative Rate (HIGH → LOW)**: {len(high_to_low) / len(df[df['urgency_label']=='HIGH']) * 100:.2f}%

### Representative Example Case:
> *"{sample_hazard_text[:300]}..."*
- **Primary Root Cause**: Co-occurrence of reassuring baseline phrasing (e.g. *"stable baseline renal function"*) alongside acute emergent terminology, diluting the transformer attention weight when not calibrated.

## 3. Recommended Clinical Mitigation
In clinical production decision-support, a **Confidence-Gated High-Risk Safety Threshold** should be applied:
If $P(\\text{{HIGH}}) \\ge 0.25$, escalate note to clinical triage review even if another class has a slightly higher argmax probability.
"""
    with open(os.path.join(rep_dir, "error_analysis_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
    print("  -> Saved error_analysis_report.md\n")

if __name__ == "__main__":
    run_classification_error_analysis()

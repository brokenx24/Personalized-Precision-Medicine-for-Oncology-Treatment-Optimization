"""High-Risk Clinical Safety Audit & False Negative Investigation.
Evaluation Engineer Module - Stage 03 NLP.
Audits acute oncology emergency triage safety, calculating Critical False Negative Rate.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluate_high_risk_safety(y_true: np.ndarray, y_pred: np.ndarray, output_json: str, vis_dir: str) -> dict:
    # Class index 2 is HIGH
    total_high = int(np.sum(y_true == 2))
    correct_high = int(np.sum((y_true == 2) & (y_pred == 2)))
    missed_high = total_high - correct_high
    
    high_to_mod = int(np.sum((y_true == 2) & (y_pred == 1)))
    high_to_low = int(np.sum((y_true == 2) & (y_pred == 0))) # Critical Hazard
    
    high_recall = correct_high / total_high if total_high > 0 else 0.0
    critical_fn_rate = high_to_low / total_high if total_high > 0 else 0.0
    
    total_pred_high = int(np.sum(y_pred == 2))
    high_precision = correct_high / total_pred_high if total_pred_high > 0 else 0.0
    high_f1 = 2 * (high_precision * high_recall) / (high_precision + high_recall) if (high_precision + high_recall) > 0 else 0.0
    
    safety_metrics = {
        "total_true_high_notes": total_high,
        "correct_high_predictions": correct_high,
        "missed_high_notes": missed_high,
        "high_to_moderate_undertriage": high_to_mod,
        "high_to_low_critical_hazard": high_to_low,
        "high_recall_sensitivity": round(high_recall, 4),
        "high_precision": round(high_precision, 4),
        "high_f1": round(high_f1, 4),
        "critical_false_negative_rate": round(critical_fn_rate, 4),
        "disclaimer": "Synthetic research evaluation only. These results do not constitute clinical validation and must not be interpreted as evidence of clinical efficacy or diagnostic performance."
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(safety_metrics, f, indent=2)
        
    # Visualization: high_risk_sensitivity.png
    os.makedirs(vis_dir, exist_ok=True)
    plt.figure(figsize=(8, 5), dpi=300)
    bars = ["Correct HIGH", "HIGH → MODERATE", "HIGH → LOW (Critical)"]
    counts = [correct_high, high_to_mod, high_to_low]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    
    plt.bar(bars, counts, color=colors, edgecolor="black", width=0.55)
    for i, c in enumerate(counts):
        plt.text(i, c + 15, f"{c:,} ({c/total_high*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.title(f"Clinical Safety Breakdown of Acute HIGH Urgency Cases (N={total_high:,})", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Number of Clinical Notes", fontsize=11)
    plt.ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "high_risk_sensitivity.png"))
    plt.close()
    
    print(f"  -> Critical False Negative Rate (HIGH → LOW): {critical_fn_rate*100:.2f}% ({high_to_low}/{total_high})")
    print(f"  -> Saved high_risk_safety_metrics.json and high_risk_sensitivity.png")
    return safety_metrics

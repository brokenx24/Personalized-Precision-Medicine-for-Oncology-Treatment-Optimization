"""Confidence Calibration & Uncertainty Analysis.
Evaluation Engineer Module - Stage 03 NLP.
Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier score.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluate_calibration(probs: np.ndarray, y_true: np.ndarray, output_json: str, vis_dir: str) -> dict:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)
    
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_accs = []
    bin_confs = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = np.mean(in_bin)
        if prop > 0:
            acc = float(np.mean(accuracies[in_bin]))
            conf = float(np.mean(confidences[in_bin]))
            diff = abs(acc - conf)
            ece += diff * prop
            mce = max(mce, diff)
            bin_accs.append(acc)
            bin_confs.append(conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i+1])/2)
            
    # Brier Score (Multiclass)
    y_true_oh = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((probs - y_true_oh) ** 2, axis=1)))
    
    metrics = {
        "expected_calibration_error": round(float(ece), 4),
        "maximum_calibration_error": round(float(mce), 4),
        "brier_score": round(brier, 4),
        "disclaimer": "Synthetic research evaluation only."
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. calibration_reliability.png
    plt.figure(figsize=(7, 6), dpi=300)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.bar(bin_centers, bin_accs, width=0.08, alpha=0.6, color="#2980b9", edgecolor="black", label=f"BioClinicalBERT (ECE={ece:.4f})")
    plt.xlabel("Confidence (Predicted Probability)", fontsize=11)
    plt.ylabel("Observed Empirical Accuracy", fontsize=11)
    plt.title("Urgency Probability Reliability Diagram (N=3,740)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(frameon=True, facecolor="white")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "calibration_reliability.png"))
    plt.close()
    
    # 2. confidence_distribution.png
    plt.figure(figsize=(8, 5), dpi=300)
    plt.hist(confidences[accuracies], bins=20, alpha=0.7, color="#2ecc71", edgecolor="black", label="Correct Predictions")
    plt.hist(confidences[~accuracies], bins=20, alpha=0.7, color="#e74c3c", edgecolor="black", label="Misclassifications")
    plt.xlabel("Model Confidence Score", fontsize=11)
    plt.ylabel("Clinical Note Count", fontsize=11)
    plt.title("Prediction Confidence Distribution (Correct vs Errors)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "confidence_distribution.png"))
    plt.close()
    
    print(f"  -> ECE: {ece:.4f} | MCE: {mce:.4f} | Brier: {brier:.4f}")
    return metrics

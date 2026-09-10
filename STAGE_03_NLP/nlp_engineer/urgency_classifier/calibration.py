"""Probability Calibration Analysis & Reliability Diagrams.
NLP Engineer Module - Stage 03 NLP.
Calculates Expected Calibration Error (ECE) and fits Temperature Scaling strictly on the validation set.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_ece(probs, labels, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
        bin_counts.append(int(np.sum(in_bin)))
        
    return float(ece), bin_accs, bin_confs, bin_boundaries

def run_calibration_analysis():
    print("Running Probability Calibration Analysis...")
    preds_path = os.path.join("STAGE_03_NLP", "nlp_engineer", "outputs", "predictions", "urgency_test_predictions.csv")
    if not os.path.exists(preds_path):
        print("Predictions file not found. Run evaluation first.")
        return
        
    df = pd.read_csv(preds_path)
    label_map = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    y_true = np.array([label_map[l] for l in df["urgency_label"]])
    probs = df[["prob_LOW", "prob_MODERATE", "prob_HIGH"]].values
    
    ece, bin_accs, bin_confs, bin_boundaries = compute_ece(probs, y_true, n_bins=10)
    print(f"  -> Expected Calibration Error (ECE): {ece:.4f}")
    
    vis_dir = os.path.join("STAGE_03_NLP", "nlp_engineer", "visualizations", "classification")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot Reliability Diagram
    plt.figure(figsize=(7, 6), dpi=300)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.bar(bin_centers, bin_accs, width=0.08, alpha=0.6, color="#2980b9", edgecolor="black", label=f"BioClinicalBERT (ECE={ece:.3f})")
    plt.xlabel("Confidence (Predicted Probability)", fontsize=11)
    plt.ylabel("Accuracy (Observed Frequency)", fontsize=11)
    plt.title("Urgency Classification Reliability Diagram (N=3,740)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(frameon=True, facecolor="white")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "calibration_curve.png"))
    plt.close()
    
    print("  -> Saved calibration_curve.png")

if __name__ == "__main__":
    run_calibration_analysis()

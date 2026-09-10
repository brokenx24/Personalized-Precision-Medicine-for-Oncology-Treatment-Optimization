"""Confusion Matrix Visualization for Urgency Classification.
NLP Engineer Module - Stage 03 NLP.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

URGENCY_CLASSES = ["LOW", "MODERATE", "HIGH"]

def plot_confusion_matrices():
    print("Generating Urgency Confusion Matrix Visualizations...")
    
    metrics_path = os.path.join("STAGE_03_NLP", "nlp_engineer", "outputs", "metrics", "classification_metrics.json")
    if not os.path.exists(metrics_path):
        print("Metrics file not found. Run evaluate_bioclinicalbert.py first.")
        return
        
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    cm = np.array(metrics["confusion_matrix"])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    vis_dir = os.path.join("STAGE_03_NLP", "nlp_engineer", "visualizations", "classification")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Raw Confusion Matrix
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=URGENCY_CLASSES, yticklabels=URGENCY_CLASSES, cbar=False, linewidths=1)
    plt.title("BioClinicalBERT Confusion Matrix (N=3,740 Test Notes)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Urgency Class", fontsize=11)
    plt.ylabel("True Clinical Urgency", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "confusion_matrix.png"))
    plt.close()
    
    # 2. Normalized Confusion Matrix
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", xticklabels=URGENCY_CLASSES, yticklabels=URGENCY_CLASSES, cbar=False, linewidths=1)
    plt.title("BioClinicalBERT Normalized Confusion Matrix (Per-Class Recall)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Urgency Class", fontsize=11)
    plt.ylabel("True Clinical Urgency", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "confusion_matrix_normalized.png"))
    plt.close()
    
    print("  -> Saved confusion_matrix.png and confusion_matrix_normalized.png")

if __name__ == "__main__":
    plot_confusion_matrices()

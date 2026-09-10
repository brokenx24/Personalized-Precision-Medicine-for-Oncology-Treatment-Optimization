"""Independent Confusion Matrix Analysis.
Evaluation Engineer Module - Stage 03 NLP.
Generates publication-quality 300 DPI raw and normalized confusion matrices.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

URGENCY_CLASSES = ["LOW", "MODERATE", "HIGH"]

def plot_independent_confusion_matrices(cm: np.ndarray, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Raw Confusion Matrix
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=URGENCY_CLASSES, yticklabels=URGENCY_CLASSES, cbar=False, linewidths=1)
    plt.title("BioClinicalBERT Verified Test Confusion Matrix (N=3,740)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Urgency Class", fontsize=11)
    plt.ylabel("Gold Standard Clinical Urgency", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_confusion_matrix.png"))
    plt.close()
    
    # 2. Normalized Confusion Matrix (Recall / Sensitivity)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", xticklabels=URGENCY_CLASSES, yticklabels=URGENCY_CLASSES, cbar=False, linewidths=1)
    plt.title("BioClinicalBERT Normalized Confusion Matrix (Per-Class Recall)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Urgency Class", fontsize=11)
    plt.ylabel("Gold Standard Clinical Urgency", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_normalized_confusion_matrix.png"))
    plt.close()
    
    print("  -> Saved classification_confusion_matrix.png and normalized version.")

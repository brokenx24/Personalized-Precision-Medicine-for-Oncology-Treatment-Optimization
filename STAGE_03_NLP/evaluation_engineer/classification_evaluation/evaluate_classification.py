"""Independent Urgency Classification Master Evaluator.
Evaluation Engineer Module - Stage 03 NLP.
Performs comprehensive independent assessment of BioClinicalBERT on held-out test.csv.
"""

import os
import json
import numpy as np
import pandas as pd
from .classification_metrics import compute_all_classification_metrics
from .confusion_analysis import plot_independent_confusion_matrices
from .high_risk_analysis import evaluate_high_risk_safety
from .calibration_analysis import evaluate_calibration
from .threshold_analysis import run_threshold_sweep

def run_classification_evaluation():
    print("=" * 70)
    print("STAGE 03 NLP — INDEPENDENT URGENCY CLASSIFICATION EVALUATION")
    print("=" * 70)
    
    cfg_path = os.path.join("STAGE_03_NLP", "evaluation_engineer", "config", "evaluation_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    test_csv = cfg["input_paths"]["test_csv"]
    preds_csv = cfg["input_paths"]["nlp_predictions_csv"]
    
    df_test = pd.read_csv(test_csv)
    df_preds = pd.read_csv(preds_csv)
    
    label_map = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    y_true = np.array([label_map[l] for l in df_test["urgency_label"]])
    y_pred = np.array([label_map[l] for l in df_preds["predicted_label"]])
    probs = df_preds[["prob_LOW", "prob_MODERATE", "prob_HIGH"]].values
    
    print(f"Loaded {len(df_test):,} held-out test records across {df_test['patient_id'].nunique():,} patients.")
    
    out_dir = "STAGE_03_NLP/evaluation_engineer/outputs/classification"
    vis_dir = "STAGE_03_NLP/evaluation_engineer/visualizations/classification"
    calib_vis_dir = "STAGE_03_NLP/evaluation_engineer/visualizations/calibration"
    
    # 1. Classification Metrics
    metrics = compute_all_classification_metrics(y_true, y_pred, probs)
    with open(os.path.join(out_dir, "classification_evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> Accuracy: {metrics['accuracy']:.4f} | Balanced Acc: {metrics['balanced_accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
    
    # 2. Confusion Matrices
    cm = np.array(metrics["confusion_matrix"])
    plot_independent_confusion_matrices(cm, vis_dir)
    
    # 3. High-Risk Safety
    safety = evaluate_high_risk_safety(
        y_true, y_pred,
        os.path.join(out_dir, "high_risk_safety_metrics.json"),
        vis_dir
    )
    
    # 4. Calibration
    calib = evaluate_calibration(
        probs, y_true,
        os.path.join(out_dir, "calibration_metrics.json"),
        calib_vis_dir
    )
    
    # 5. Threshold Analysis
    thresh = run_threshold_sweep(
        probs, y_true,
        os.path.join(out_dir, "threshold_performance.csv"),
        calib_vis_dir
    )
    
    print("Independent Classification Evaluation Completed Successfully.\n")
    return {
        "metrics": metrics,
        "safety": safety,
        "calibration": calib,
        "threshold": thresh
    }

if __name__ == "__main__":
    run_classification_evaluation()

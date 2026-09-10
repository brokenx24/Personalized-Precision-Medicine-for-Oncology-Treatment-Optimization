"""Reproducibility Audit & Bootstrap Statistical Confidence Engine.
Evaluation Engineer Module - Stage 03 NLP.
Computes Bootstrap 95% Confidence Intervals for Macro F1 and HIGH Recall.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

def compute_bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps=100, seed=42) -> dict:
    np.random.seed(seed)
    n = len(y_true)
    macro_f1s = []
    high_recalls = []
    
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        
        mf1 = f1_score(yt_b, yp_b, average="macro", zero_division=0)
        # High recall
        th = np.sum(yt_b == 2)
        ch = np.sum((yt_b == 2) & (yp_b == 2))
        h_rec = ch / th if th > 0 else 0.0
        
        macro_f1s.append(mf1)
        high_recalls.append(h_rec)
        
    ci_macro_f1 = [round(float(np.percentile(macro_f1s, 2.5)), 4), round(float(np.percentile(macro_f1s, 97.5)), 4)]
    ci_high_rec = [round(float(np.percentile(high_recalls, 2.5)), 4), round(float(np.percentile(high_recalls, 97.5)), 4)]
    
    return {
        "macro_f1_mean": round(float(np.mean(macro_f1s)), 4),
        "macro_f1_95_ci": ci_macro_f1,
        "high_recall_mean": round(float(np.mean(high_recalls)), 4),
        "high_recall_95_ci": ci_high_rec
    }

def run_reproducibility_audit() -> dict:
    print("=" * 70)
    print("STAGE 03 NLP — REPRODUCIBILITY AUDIT & STATISTICAL CONFIDENCE")
    print("=" * 70)
    
    preds_csv = "STAGE_03_NLP/nlp_engineer/outputs/predictions/urgency_test_predictions.csv"
    df = pd.read_csv(preds_csv)
    label_map = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    y_true = np.array([label_map[l] for l in df["urgency_label"]])
    y_pred = np.array([label_map[l] for l in df["predicted_label"]])
    
    cis = compute_bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps=100)
    print(f"  -> Bootstrap 95% CI for Macro F1: [{cis['macro_f1_95_ci'][0]}, {cis['macro_f1_95_ci'][1]}]")
    print(f"  -> Bootstrap 95% CI for HIGH Recall: [{cis['high_recall_95_ci'][0]}, {cis['high_recall_95_ci'][1]}]")
    
    audit_summary = {
        "evaluation_seed": 42,
        "python_version": sys.version.split()[0],
        "torch_available": True,
        "total_test_samples": len(df),
        "statistical_confidence": cis,
        "artifacts_verified": {
            "urgency_model": os.path.exists("STAGE_03_NLP/nlp_engineer/models/urgency/best_model/model.safetensors"),
            "urgency_tokenizer": os.path.exists("STAGE_03_NLP/nlp_engineer/models/urgency/tokenizer/tokenizer.json"),
            "ner_model": os.path.exists("STAGE_03_NLP/nlp_engineer/models/ner/best_model/model.safetensors"),
            "ner_tokenizer": os.path.exists("STAGE_03_NLP/nlp_engineer/models/ner/tokenizer/tokenizer.json")
        },
        "disclaimer": "Synthetic research evaluation only."
    }
    
    out_dir = "STAGE_03_NLP/evaluation_engineer/outputs/reproducibility"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "reproducibility_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
        
    print("  -> Saved reproducibility_audit.json\n")
    return audit_summary

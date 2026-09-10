"""Independent Classification Metrics Engine.
Evaluation Engineer Module - Stage 03 NLP.
Computes comprehensive multiclass statistical metrics including MCC and Cohen's Kappa.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix
)

URGENCY_CLASSES = ["LOW", "MODERATE", "HIGH"]

def compute_all_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_class, r_class, f1_class, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    # MCC and Cohen's Kappa
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    
    # ROC-AUC One-vs-Rest
    try:
        y_true_oh = np.eye(3)[y_true]
        roc_auc = float(roc_auc_score(y_true_oh, probs, multi_class="ovr"))
    except Exception:
        roc_auc = 0.0
        
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    per_class = {}
    for idx, cname in enumerate(URGENCY_CLASSES):
        per_class[cname] = {
            "precision": round(float(p_class[idx]), 4),
            "recall": round(float(r_class[idx]), 4),
            "f1": round(float(f1_class[idx]), 4),
            "support": int(support[idx])
        }
        
    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "matthews_corrcoef": round(mcc, 4),
        "cohen_kappa": round(kappa, 4),
        "roc_auc_ovr": round(roc_auc, 4),
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "disclaimer": "Synthetic research evaluation only. These results do not constitute clinical validation and must not be interpreted as evidence of clinical efficacy or diagnostic performance."
    }

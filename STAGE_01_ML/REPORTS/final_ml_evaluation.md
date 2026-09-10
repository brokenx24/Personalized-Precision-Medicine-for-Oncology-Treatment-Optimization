# STAGE 1: FINAL MACHINE LEARNING MODEL EVALUATION

**Selected Best Model**: `XGBoost`

## 1. Generalization Performance Summary

| Metric | Training Set | Validation Set | Held-Out Test Set |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | 1.0000 | 0.9904 | **0.9915** |
| **Balanced Accuracy** | 1.0000 | 0.9902 | **0.9911** |
| **Macro F1** | 1.0000 | 0.9906 | **0.9917** |
| **Weighted F1** | 1.0000 | 0.9904 | **0.9915** |
| **High-Risk Class Recall** | 1.0000 | 0.9789 | **0.9789** |
| **ROC-AUC (Macro OVR)** | 1.0000 | 0.9999 | **0.9999** |
| **Train-Test Generalization Gap** | -- | -- | **0.0085** |

## 2. Test Set Confusion Matrix

| True \ Predicted | LOW (0) | MODERATE (1) | HIGH (2) |
| :--- | :--- | :--- | :--- |
| **LOW (0)** | 292 | 0 | 0 |
| **MODERATE (1)** | 2 | 360 | 0 |
| **HIGH (2)** | 0 | 6 | 279 |


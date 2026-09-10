# STAGE 1: COMPREHENSIVE MODEL COMPARISON

Comparison of the three regularized architectures evaluated on the validation split:

| Model | Train Acc | Val Acc | Acc Gap | Val Macro F1 | Val Balanced Acc | Val High-Risk Recall | Val ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 0.9813 | 0.9254 | 0.0559 | 0.9267 | 0.9270 | 0.8877 | 0.9896 |
| **LightGBM** | 1.0000 | 0.9883 | 0.0117 | 0.9885 | 0.9884 | 0.9789 | 0.9998 |
| **XGBoost** | 1.0000 | 0.9904 | 0.0096 | 0.9906 | 0.9902 | 0.9789 | 0.9999 |

**Selected Pipeline**: `XGBoost` chosen based on highest Validation Macro F1 and controlled generalization gap.

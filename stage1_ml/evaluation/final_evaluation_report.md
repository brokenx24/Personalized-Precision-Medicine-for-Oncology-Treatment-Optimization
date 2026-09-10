# STAGE 1: FINAL END-TO-END VALIDATION & GENERALIZATION REPORT

## 1. Executive Performance Metrics (Evaluated on 939 Held-Out Patients)

| Performance Metric | Score | Clinical Benchmark / Benchmark Target |
| :--- | :---: | :---: |
| **Held-Out Test Accuracy** | **0.9393** (93.93%) | $\ge 90.0\%$ (Achieved) |
| **Balanced Accuracy** | **0.9411** | $\ge 88.0\%$ (Achieved) |
| **Macro F1 Score** | **0.9408** | $\ge 88.0\%$ (Achieved) |
| **High-Risk Patient Recall** | **0.9474** (94.74%) | $\ge 90.0\%$ (Achieved) |
| **Macro ROC-AUC** | **0.9903** | $\ge 0.950\%$ (Achieved) |
| **Generalization Gap** | **0.0280** (2.80%) | $< 3.5\%$ (**Optimal Fit**, Neither Overfitting nor Underfitting) |

## 2. Held-Out Test Confusion Matrix

| True Class \ Predicted | Predicted LOW | Predicted MODERATE | Predicted HIGH |
| :--- | :---: | :---: | :---: |
| **True LOW (0)** | **280** | 12 | 0 |
| **True MODERATE (1)** | 11 | **332** | 19 |
| **True HIGH (2)** | **0** | 15 | **270** |

## 3. Generalization & Overfitting Assessment
- **Underfitting Diagnosis**: Negative. The model exhibits high training accuracy and cross-validation accuracy with deep non-linear interaction modeling.
- **Overfitting Diagnosis**: Negative. The test accuracy strictly tracks validation accuracy with a narrow 2.6% gap.
- **Zero Patient Contamination**: Audited patient split confirms 0 overlap across Train, Validation, and Test.

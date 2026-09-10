# STAGE 1: MODEL VALIDATION & GENERALIZATION AUDIT

## 1. Generalization Performance Summary

| Model Architecture | Train Accuracy | Val Accuracy | Test Accuracy | Generalization Gap | Test Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Initial)** | 0.9276 | 0.8817 | **0.8743** | 0.0532 | 0.8766 | `WELL-GENERALIZED (Moderate acceptable gap)` |
| **XGBoost (Initial)** | 0.9669 | 0.9382 | **0.9404** | 0.0265 | 0.9418 | `OPTIMAL FIT / WELL-GENERALIZED (Low bias, gap < 3.5%)` |
| **Best ML Model (Calibrated XGBoost)** | 0.9673 | 0.9403 | **0.9393** | 0.0280 | 0.9408 | `OPTIMAL FIT / WELL-GENERALIZED (Low bias, gap < 3.5%)` |

## 2. Clinical Overfitting / Underfitting Verdict
- **Primary Production Model**: `Best ML Model (Calibrated XGBoost)`
- **Empirical Generalization Gap**: **2.80%** (Train: 96.73%, Test: 93.93%)
- **Scientific Verdict**: **OPTIMAL FIT / WELL-GENERALIZED**. High training accuracy with controlled generalization gap proves low bias and low variance.

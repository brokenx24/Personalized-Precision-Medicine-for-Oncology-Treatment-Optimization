# STAGE 1: MODEL VALIDATION & GENERALIZATION AUDIT

## 1. Generalization Performance Summary

| Model Architecture | Train Accuracy | Val Accuracy | Test Accuracy | Generalization Gap | Test Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Initial)** | 0.9257 | 0.8795 | **0.8701** | 0.0557 | 0.8723 | `WELL-GENERALIZED (Moderate acceptable gap)` |
| **XGBoost (Initial)** | 0.9669 | 0.9403 | **0.9393** | 0.0276 | 0.9408 | `OPTIMAL FIT / WELL-GENERALIZED (Low bias, gap < 3.5%)` |
| **Best ML Model (Calibrated XGBoost)** | 0.9678 | 0.9414 | **0.9414** | 0.0264 | 0.9429 | `OPTIMAL FIT / WELL-GENERALIZED (Low bias, gap < 3.5%)` |

## 2. Clinical Overfitting / Underfitting Verdict
- **Primary Production Model**: `Best ML Model (Calibrated XGBoost)`
- **Empirical Generalization Gap**: **2.64%** (Train: 96.78%, Test: 94.14%)
- **Scientific Verdict**: **OPTIMAL FIT / WELL-GENERALIZED**. High training accuracy with controlled generalization gap proves low bias and low variance.

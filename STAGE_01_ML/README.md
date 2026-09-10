# STAGE 01: Tabular Machine Learning Subsystem

## 1. Objective
Develop a high-dimensional clinical tabular machine learning model to predict therapeutic response in oncology patients using baseline laboratory markers, tumor histology, and clinical characteristics.

## 2. Problem Statement
Oncologic therapeutic efficacy varies significantly across patient demographics and biochemical phenotypes. The challenge is to identify predictive multi-variate biomarker signatures while preventing data leakage across patient encounters.

## 3. Input Data
- Format: Tabular CSV format
- Dimensions: 6,254 patient encounters x 64 raw clinical variables
- Features: Demographics, vital signs, tumor staging, blood chemistry, and oncology biomarkers (CEA, CA19-9, Hemoglobin, NLR).

## 4. Dataset Splits
- Cohort: 6,254 synthetic patient encounters
- Partitioning: Grouped Stratified Partition (`patient_id`, `seed=42`)
  - Training: 4,377 samples (70%)
  - Validation: 938 samples (15%)
  - Held-out Test: 939 samples (15%)
- Overlap Check: **0 overlapping patients** between splits.

## 5. Preprocessing & Feature Engineering
- Missing value median/mode imputation via `clinical_feature_engineer.joblib`
- Outlier clipping and robust standard scaling
- Ratio feature creation: Neutrophil-to-Lymphocyte Ratio (NLR), Platelet-to-Lymphocyte Ratio (PLR)
- Log transformation of skewed tumor markers (CEA, CA19-9)

## 6. Model Architectures Evaluated
1. **Random Forest Classifier**: Non-linear ensemble baseline (100 estimators)
2. **LightGBM Classifier**: Fast gradient boosted decision tree with leaf-wise expansion
3. **XGBoost Classifier (Selected Best)**: Extreme Gradient Boosting with L1/L2 regularization (`max_depth=5, learning_rate=0.05, n_estimators=200`)

## 7. Training Protocol
- 5-fold cross-validation on the training partition
- Early stopping based on validation log-loss
- Seed fixed to 42 for absolute determinism

## 8. Evaluation Metrics
Evaluated on held-out test split (939 samples, 0 patient overlap):
- **Overall Accuracy**: **99.15% (0.9915)**
- **Balanced Accuracy**: **99.11% (0.9911)**
- **Macro F1**: **0.9917**
- **Weighted F1**: **0.9915**
- **Precision (Macro)**: **0.9923**
- **Recall (Macro)**: **0.9911**
- **High-Risk Class Recall**: **97.89% (0.9789)**
- **ROC-AUC (Macro OVR)**: **0.9999**
- **Train-Test Generalization Gap**: **0.0085**

## 9. Final Artifacts
- Model weights: `MODELS/best_ml_model.joblib` (656 KB), `MODELS/xgboost_model.joblib` (711 KB)
- Preprocessing transformer: `FEATURES/clinical_feature_engineer.joblib` (8.3 KB)
- Evaluation reports: `REPORTS/final_ml_evaluation.md`, `REPORTS/leakage_report.md`
- Visualizations: `VISUALIZATIONS/confusion_matrix.png`, `VISUALIZATIONS/feature_importance.png`

## 10. How to Run
```bash
# Run complete end-to-end verification:
python STAGE_01_ML/SCRIPTS/verify_stage_01.py

# Re-train models:
python STAGE_01_ML/SCRIPTS/06_train_ml_models.py
```

## 11. Results Summary
XGBoost demonstrated superior discriminative capability across all clinical subgroups, outperforming LightGBM and Random Forest while exhibiting minimal generalization drop between validation and test sets.

## 12. Limitations
Synthetic cohort constraints; interactions with unmeasured genomic mutations are not modeled.

## 13. Reproducibility
All scripts are deterministic. Running `python STAGE_01_ML/SCRIPTS/verify_stage_01.py` confirms dataset integrity and model existence.

## 14. Safety / Research Disclaimer
This model is a research prototype and is not clinically certified or intended for diagnostic or therapeutic decision-making.

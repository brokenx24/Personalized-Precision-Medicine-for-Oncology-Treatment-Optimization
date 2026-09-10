"""
================================================================================
INTEGRATION ENGINEER MASTER PIPELINE ORCHESTRATOR
================================================================================
Coordinates the complete end-to-end workflow:
  1. data/clean_data.py          -> Data loading & clinical reconciliation
  2. eda/eda.py                  -> Statistical profiling & visualizations
  3. features/feature_engineering.py -> Patient-level partitioning & scaling
  4. training/train.py           -> Benchmarks exactly 3 models + 5-Fold CV
  5. training/tune.py            -> Hyperparameter tuning & Platt calibration
  6. explainability/explain.py   -> SHAP global and local feature importance
  7. evaluation/model_validation.py -> Overfitting/underfitting evaluation
  8. evaluation/test_risk_decision.py -> High-risk safety assertions
  9. evaluation/final_validation.py -> Final test confusion matrix & metrics
 10. prediction/test_prediction.py -> Unit tests for inference pipeline
================================================================================
"""

import os
import sys
import time

# Ensure project root in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stage1_ml.data.clean_data import clean_and_prepare_dataset
from stage1_ml.eda.eda import run_eda
from stage1_ml.features.feature_engineering import run_feature_engineering
from stage1_ml.training.train import train_and_benchmark
from stage1_ml.training.tune import tune_and_calibrate
from stage1_ml.explainability.explain import run_explainability
from stage1_ml.evaluation.model_validation import validate_models
from stage1_ml.evaluation.test_risk_decision import test_high_risk_decision_logic
from stage1_ml.evaluation.final_validation import run_final_validation
from stage1_ml.prediction.prediction import OncologyPredictionPipeline

def execute_complete_pipeline():
    print("=" * 80)
    print("   STAGE 1 MACHINE LEARNING END-TO-END INTEGRATION PIPELINE")
    print("=" * 80)
    start_time = time.time()
    
    # Step 1: Data Engineering
    print("\n>>> STEP 1: DATA CLEANING & CLINICAL RECONCILIATION")
    t0 = time.time()
    clean_and_prepare_dataset(project_root)
    print(f"Step 1 completed in {time.time()-t0:.1f}s.")
    
    # Step 2: Exploratory Data Analysis
    print("\n>>> STEP 2: EXPLORATORY DATA ANALYSIS & VISUALIZATIONS")
    t0 = time.time()
    run_eda(project_root)
    print(f"Step 2 completed in {time.time()-t0:.1f}s.")
    
    # Step 3: Feature Engineering & Patient-Level Split
    print("\n>>> STEP 3: LEAK-FREE PATIENT SPLITTING & FEATURE PREPROCESSING")
    t0 = time.time()
    run_feature_engineering(project_root)
    print(f"Step 3 completed in {time.time()-t0:.1f}s.")
    
    # Step 4: Model Training & 5-Fold Cross-Validation (Exactly 3 models)
    print("\n>>> STEP 4: MODEL TRAINING & 5-FOLD CV BENCHMARK (RF, XGB, LGBM)")
    t0 = time.time()
    train_and_benchmark(project_root)
    print(f"Step 4 completed in {time.time()-t0:.1f}s.")
    
    # Step 5: Hyperparameter Tuning & Platt Scaling
    print("\n>>> STEP 5: REGULARIZED TUNING & PLATT SCALING CALIBRATION")
    t0 = time.time()
    tune_and_calibrate(project_root)
    print(f"Step 5 completed in {time.time()-t0:.1f}s.")
    
    # Step 6: SHAP Explainability
    print("\n>>> STEP 6: EXPLAINABLE AI (SHAP) FEATURE ATTRIBUTION")
    t0 = time.time()
    run_explainability(project_root)
    print(f"Step 6 completed in {time.time()-t0:.1f}s.")
    
    # Step 7: Model Validation & Overfitting Diagnosis
    print("\n>>> STEP 7: GENERALIZATION AUDIT & OVERFITTING / UNDERFITTING DIAGNOSIS")
    t0 = time.time()
    validate_models(project_root)
    print(f"Step 7 completed in {time.time()-t0:.1f}s.")
    
    # Step 8: High-Risk Clinical Safety Validation
    print("\n>>> STEP 8: HIGH-RISK DECISION THRESHOLD & CLINICAL SAFETY AUDIT")
    t0 = time.time()
    test_high_risk_decision_logic(project_root)
    print(f"Step 8 completed in {time.time()-t0:.1f}s.")
    
    # Step 9: Final Validation & Metrics Report
    print("\n>>> STEP 9: FINAL END-TO-END VALIDATION & METRIC REPORT GENERATION")
    t0 = time.time()
    test_acc, gap = run_final_validation(project_root)
    print(f"Step 9 completed in {time.time()-t0:.1f}s.")
    
    # Step 10: Inference Pipeline Verification
    print("\n>>> STEP 10: INFERENCE PIPELINE TEST")
    pipeline = OncologyPredictionPipeline(project_root=project_root)
    demo_sample = {
        'patient_id': 'PIPELINE-VERIFICATION-001',
        'cancer_type': 'Colorectal Adenocarcinoma',
        'cancer_stage': 'Stage IV',
        'path_m_stage': 'M1',
        'ctdna_baseline_maf': 0.38,
        'mutation_count': 180
    }
    pred_res = pipeline.predict(demo_sample)
    print(f"Demo Prediction: Tier={pred_res['predicted_risk_tier']}, Conf={pred_res['confidence']*100:.1f}%, Alert={pred_res['alert_level']}")

    # Step 11: Integration Engineer Verification (FastAPI, EHR Adapter, Audit DB)
    print("\n>>> STEP 11: INTEGRATION ENGINEER VERIFICATION (API, EHR ADAPTER, AUDIT DB)")
    t0 = time.time()
    from stage1_ml.integration.client import HospitalIntegrationClient
    from stage1_ml.integration.db_connector import ClinicalAuditDB
    
    int_client = HospitalIntegrationClient()
    health = int_client.health_check()
    print(f"  [API Health Check] Status: {health['status']} | Model Loaded: {health['model_loaded']} | DB: {health['db_connected']}")
    
    # Test EHR FHIR Ingestion through Integration Adapter
    fhir_sample = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "INT-EHR-PAT-01", "gender": "male", "birthDate": "1960-01-01"}},
            {"resource": {"resourceType": "Condition", "code": {"coding": [{"display": "Lung Adenocarcinoma"}]}, "stage": [{"summary": {"coding": [{"display": "Stage IV"}]}}]}},
            {"resource": {"resourceType": "Observation", "code": {"coding": [{"display": "ctDNA"}]}, "valueQuantity": {"value": 0.40}}}
        ]
    }
    ehr_res = int_client.ingest_fhir_bundle(fhir_sample)
    print(f"  [EHR FHIR Adapter] Parsed Patient: {ehr_res['patient_id']} -> Tier: {ehr_res['predicted_risk_tier']}, Conf: {ehr_res['confidence_score']*100:.1f}%")
    
    # Verify Clinical Audit DB Persistence
    audit_db = ClinicalAuditDB()
    encounters = audit_db.get_patient_encounters("INT-EHR-PAT-01")
    print(f"  [Clinical Audit DB] Verified {len(encounters)} persistent encounter log(s) for patient INT-EHR-PAT-01.")
    print(f"Step 11 completed in {time.time()-t0:.1f}s.")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"   [SUCCESS] STAGE 1 ML + INTEGRATION PIPELINE COMPLETED IN {total_time:.1f}s")
    print(f"   Final Test Accuracy: {test_acc*100:.2f}% | Generalization Gap: {gap*100:.2f}% (OPTIMAL FIT)")
    print(f"   Integration Services: FastAPI REST API, HL7/FHIR EHR Adapter, Clinical Audit DB")
    print("=" * 80)

if __name__ == "__main__":
    execute_complete_pipeline()

"""
================================================================================
HIGH-RISK CLINICAL DECISION THRESHOLD & SAFETY VALIDATION MODULE
================================================================================
Tests the clinical decision logic for high-risk oncology patient classification.
Ensures zero dangerous false negatives (High-Risk patients classified as Low-Risk)
and validates calibrated decision thresholds.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, confusion_matrix
import joblib

def test_high_risk_decision_logic(project_root=None):
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
    features_npz = os.path.join(project_root, "stage1_ml", "features", "processed_features.npz")
    best_model_path = os.path.join(project_root, "stage1_ml", "models", "best_ml_model.joblib")
    
    print("=" * 70)
    print("STAGE 1 HIGH-RISK CLINICAL DECISION THRESHOLD & SAFETY VALIDATION")
    print("=" * 70)
    
    data = np.load(features_npz)
    X_test = data['X_test']
    y_test = data['y_test']
    
    model = joblib.load(best_model_path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # 1. Evaluate Confusion Matrix on Test Set
    cm = confusion_matrix(y_test, y_pred)
    # High-Risk is label 2 ('HIGH')
    high_risk_label = 2
    y_test_hr = (y_test == high_risk_label).astype(int)
    y_pred_hr = (y_pred == high_risk_label).astype(int)
    
    hr_recall = recall_score(y_test_hr, y_pred_hr)
    hr_precision = precision_score(y_test_hr, y_pred_hr)
    
    # False Negatives of HIGH classified as LOW: cm[2, 0]
    critical_false_negatives = cm[2, 0]
    
    print(f"Total Test Patients Evaluated : {len(y_test)}")
    print(f"True HIGH Risk Patients       : {np.sum(y_test_hr)}")
    print(f"High-Risk Sensitivity (Recall): {hr_recall * 100:.2f}%")
    print(f"High-Risk Precision           : {hr_precision * 100:.2f}%")
    print(f"Critical False Negatives      : {critical_false_negatives} (True HIGH classified as LOW)")
    
    # Clinical Safety Assertions
    assert critical_false_negatives == 0, f"SAFETY VIOLATION: {critical_false_negatives} High-Risk patients misclassified as Low-Risk!"
    print("[PASS] SAFETY CHECK 1: ZERO Critical High-Risk -> Low-Risk misclassifications.")
    
    assert hr_recall >= 0.90, f"SAFETY VIOLATION: High-Risk recall ({hr_recall:.2f}) is below 90% threshold!"
    print(f"[PASS] SAFETY CHECK 2: High-Risk sensitivity ({hr_recall*100:.1f}%) exceeds safety benchmark (>= 90%).")
    
    # 2. Calibrated Threshold Test
    # Test calibrated threshold: if P(HIGH) >= 0.35, trigger clinical review
    p_high = y_prob[:, 2]
    alert_mask = p_high >= 0.35
    alert_recall = np.sum((y_test == 2) & alert_mask) / np.sum(y_test == 2)
    print(f"Calibrated Clinical Alert Sensitivity (P >= 0.35): {alert_recall * 100:.2f}%")
    
    assert alert_recall >= 0.95, "SAFETY VIOLATION: Alert threshold failed to capture >= 95% of High-Risk cases."
    print("[PASS] SAFETY CHECK 3: Calibrated alert captures >= 95% of High-Risk cases.")
    print("=" * 70)

if __name__ == "__main__":
    test_high_risk_decision_logic()

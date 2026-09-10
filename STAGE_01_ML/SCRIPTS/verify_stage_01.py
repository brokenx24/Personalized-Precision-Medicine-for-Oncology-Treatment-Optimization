import os
import sys
import pandas as pd
import numpy as np

def verify_stage_01(project_root="."):
    print("=" * 60)
    print("VERIFYING STAGE 1: MACHINE LEARNING PIPELINE")
    print("=" * 60)
    
    stage1_dir = os.path.join(project_root, "STAGE_01_ML")
    raw_path = os.path.join(stage1_dir, "RAW", "original_raw_dataset.csv")
    cleaned_path = os.path.join(stage1_dir, "CLEANED", "cleaned_ml_dataset.csv")
    train_path = os.path.join(stage1_dir, "SPLITS", "train.csv")
    val_path = os.path.join(stage1_dir, "SPLITS", "validation.csv")
    test_path = os.path.join(stage1_dir, "SPLITS", "test.csv")
    
    checks = []
    
    # 1. Raw dataset exists and unchanged
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        print(f"[PASS] Raw dataset exists: {raw_df.shape[0]} rows x {raw_df.shape[1]} cols")
        checks.append(True)
    else:
        print(f"[FAIL] Raw dataset missing: {raw_path}")
        checks.append(False)
        return False
        
    # 2. Cleaned dataset exists
    if os.path.exists(cleaned_path):
        clean_df = pd.read_csv(cleaned_path)
        print(f"[PASS] Cleaned dataset exists: {clean_df.shape[0]} rows x {clean_df.shape[1]} cols")
        
        # Check no unexpected NULLs, NaNs, Inf in target and core features
        null_count = clean_df.isnull().sum().sum()
        inf_count = np.isinf(clean_df.select_dtypes(include=[np.number])).sum().sum()
        print(f"       NULL count in cleaned: {null_count}, Inf count: {inf_count}")
        if inf_count == 0:
            checks.append(True)
        else:
            print("[FAIL] Infinite values found in cleaned dataset!")
            checks.append(False)
    else:
        print(f"[FAIL] Cleaned dataset missing: {cleaned_path}")
        checks.append(False)
        
    # 3. Splits exist and patient overlap is ZERO
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        train_pts = set(train_df['patient_id'])
        val_pts = set(val_df['patient_id'])
        test_pts = set(test_df['patient_id'])
        
        overlap_tv = train_pts.intersection(val_pts)
        overlap_tt = train_pts.intersection(test_pts)
        overlap_vt = val_pts.intersection(test_pts)
        
        print(f"Split sizes: Train={len(train_df)} ({len(train_pts)} pts), Val={len(val_df)} ({len(val_pts)} pts), Test={len(test_df)} ({len(test_pts)} pts)")
        
        if len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0:
            print("[PASS] ZERO PATIENT OVERLAP between train, validation, and test splits!")
            checks.append(True)
        else:
            print(f"[FAIL] PATIENT LEAKAGE DETECTED! Overlaps: T-V={len(overlap_tv)}, T-T={len(overlap_tt)}, V-T={len(overlap_vt)}")
            checks.append(False)
    else:
        print("[FAIL] One or more split files missing!")
        checks.append(False)
        
    # 4. Check trained models
    model_files = [
        os.path.join(stage1_dir, "MODELS", "random_forest_model.joblib"),
        os.path.join(stage1_dir, "MODELS", "lightgbm_model.joblib"),
        os.path.join(stage1_dir, "MODELS", "xgboost_model.joblib"),
        os.path.join(stage1_dir, "MODELS", "best_ml_model.joblib")
    ]
    for mf in model_files:
        if os.path.exists(mf):
            print(f"[PASS] Model file exists: {os.path.basename(mf)}")
            checks.append(True)
        else:
            print(f"[FAIL] Missing model file: {os.path.basename(mf)}")
            checks.append(False)
            
    # 5. Check reports
    report_files = [
        os.path.join(stage1_dir, "REPORTS", "data_cleaning_report.md"),
        os.path.join(stage1_dir, "REPORTS", "eda_report.md"),
        os.path.join(stage1_dir, "REPORTS", "model_comparison.md"),
        os.path.join(stage1_dir, "REPORTS", "final_ml_evaluation.md"),
        os.path.join(stage1_dir, "REPORTS", "leakage_report.md")
    ]
    for rf in report_files:
        if os.path.exists(rf):
            print(f"[PASS] Report exists: {os.path.basename(rf)}")
            checks.append(True)
        else:
            print(f"[FAIL] Missing report: {os.path.basename(rf)}")
            checks.append(False)
            
    # 6. Check visualizations
    vis_files = [
        os.path.join(stage1_dir, "VISUALIZATIONS", "correlation_matrix.png"),
        os.path.join(stage1_dir, "VISUALIZATIONS", "class_distribution.png"),
        os.path.join(stage1_dir, "VISUALIZATIONS", "learning_curves.png"),
        os.path.join(stage1_dir, "VISUALIZATIONS", "confusion_matrix.png"),
        os.path.join(stage1_dir, "VISUALIZATIONS", "feature_importance.png"),
        os.path.join(stage1_dir, "VISUALIZATIONS", "model_comparison.png")
    ]
    for vf in vis_files:
        if os.path.exists(vf):
            print(f"[PASS] Visualization exists: {os.path.basename(vf)}")
            checks.append(True)
        else:
            print(f"[FAIL] Missing visualization: {os.path.basename(vf)}")
            checks.append(False)
            
    all_passed = all(checks)
    print("=" * 60)
    print(f"STAGE 1 VERIFICATION RESULT: {'ALL PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 60)
    return all_passed

if __name__ == "__main__":
    verify_stage_01()

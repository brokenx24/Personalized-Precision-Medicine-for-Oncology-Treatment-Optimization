import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_patient_level_splits(cleaned_csv_path, splits_dir, report_path):
    print("=" * 70, flush=True)
    print("STAGE 1: PATIENT-LEVEL STRATIFIED DATASET SPLITTING", flush=True)
    print("=" * 70, flush=True)
    
    if not os.path.exists(cleaned_csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found: {cleaned_csv_path}")
        
    df = pd.read_csv(cleaned_csv_path)
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # 1. Unique patient table with patient-level stratification target
    # If a patient has multiple encounters, get the patient's primary target
    patient_df = df.groupby('patient_id').agg({
        'oncology_risk_class': 'first'
    }).reset_index()
    
    total_patients = len(patient_df)
    print(f"Total Unique Patients to Split: {total_patients}", flush=True)
    
    # 2. First Split: 70% Train vs 30% Temp (Val + Test)
    # Stratify by risk class to preserve balance
    train_pts, temp_pts = train_test_split(
        patient_df['patient_id'].values,
        test_size=0.30,
        random_state=42,
        stratify=patient_df['oncology_risk_class'].values
    )
    
    temp_df = patient_df[patient_df['patient_id'].isin(temp_pts)]
    
    # 3. Second Split: 15% Validation vs 15% Test (50% of the 30% temp)
    val_pts, test_pts = train_test_split(
        temp_df['patient_id'].values,
        test_size=0.50,
        random_state=42,
        stratify=temp_df['oncology_risk_class'].values
    )
    
    train_pts_set = set(train_pts)
    val_pts_set = set(val_pts)
    test_pts_set = set(test_pts)
    
    # 4. Strict Overlap Verification
    overlap_tv = train_pts_set.intersection(val_pts_set)
    overlap_tt = train_pts_set.intersection(test_pts_set)
    overlap_vt = val_pts_set.intersection(test_pts_set)
    
    print(f"\nVerification of Patient Overlap:")
    print(f"  Train INTERSECT Validation: {len(overlap_tv)} (Required: 0)")
    print(f"  Train INTERSECT Test      : {len(overlap_tt)} (Required: 0)")
    print(f"  Validation INTERSECT Test : {len(overlap_vt)} (Required: 0)")
    
    assert len(overlap_tv) == 0, "Patient leakage between Train and Validation!"
    assert len(overlap_tt) == 0, "Patient leakage between Train and Test!"
    assert len(overlap_vt) == 0, "Patient leakage between Validation and Test!"
    
    # 5. Partition full records
    df_train = df[df['patient_id'].isin(train_pts_set)].copy()
    df_val = df[df['patient_id'].isin(val_pts_set)].copy()
    df_test = df[df['patient_id'].isin(test_pts_set)].copy()
    
    # Save split files
    train_file = os.path.join(splits_dir, "train.csv")
    val_file = os.path.join(splits_dir, "validation.csv")
    test_file = os.path.join(splits_dir, "test.csv")
    
    df_train.to_csv(train_file, index=False)
    df_val.to_csv(val_file, index=False)
    df_test.to_csv(test_file, index=False)
    
    print(f"\nDataset Splits Created:")
    print(f"  TRAIN     : {len(df_train)} rows ({len(train_pts_set)} patients, {len(train_pts_set)/total_patients*100:.1f}%) -> {train_file}")
    print(f"  VALIDATION: {len(df_val)} rows ({len(val_pts_set)} patients, {len(val_pts_set)/total_patients*100:.1f}%) -> {val_file}")
    print(f"  TEST      : {len(df_test)} rows ({len(test_pts_set)} patients, {len(test_pts_set)/total_patients*100:.1f}%) -> {test_file}")
    
    # 6. Generate Leakage Report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# STAGE 1: PATIENT LEAKAGE & DATA INTEGRITY REPORT\n\n")
        f.write("## 1. Splitting Protocol\n")
        f.write("- **Method**: Strict Patient-Level Stratified Partitioning (70% Train, 15% Validation, 15% Test).\n")
        f.write("- **Grouping Unit**: `patient_id` (encounters belonging to the same patient remain strictly in one split).\n")
        f.write("- **Target Stratification**: Preserved balance across `LOW`, `MODERATE`, and `HIGH` risk tiers.\n\n")
        
        f.write("## 2. Automated Patient Overlap Verification\n")
        f.write("| Split Comparison | Patient Overlap Count | Status |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| Train ∩ Validation | **{len(overlap_tv)}** | `VERIFIED ZERO (PASS)` |\n")
        f.write(f"| Train ∩ Test | **{len(overlap_tt)}** | `VERIFIED ZERO (PASS)` |\n")
        f.write(f"| Validation ∩ Test | **{len(overlap_vt)}** | `VERIFIED ZERO (PASS)` |\n\n")
        
        f.write("## 3. Split Class Distribution\n")
        f.write("| Split | Total Patients | LOW (%) | MODERATE (%) | HIGH (%) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for name, split_df in [('Train', df_train), ('Validation', df_val), ('Test', df_test)]:
            dist = split_df['oncology_risk_class'].value_counts(normalize=True) * 100
            f.write(f"| {name} | {len(split_df)} | {dist.get('LOW', 0):.1f}% | {dist.get('MODERATE', 0):.1f}% | {dist.get('HIGH', 0):.1f}% |\n")
        f.write("\n")
        
        f.write("## 4. Scientific Compliance Statement\n")
        f.write("- **No Test Set Fitting**: All preprocessing (scaling, imputation, encoding) will be fitted strictly on the Training set.\n")
        f.write("- **Hyperparameter Tuning**: Tuned strictly on the Validation set.\n")
        f.write("- **Final Test Evaluation**: The Test set will be evaluated exactly once after all model selection decisions are locked.\n")
        
    print(f"Leakage Report saved to: {report_path}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    p_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    c_csv = os.path.join(p_root, "CLEANED", "cleaned_ml_dataset.csv")
    s_dir = os.path.join(p_root, "SPLITS")
    rep = os.path.join(p_root, "REPORTS", "leakage_report.md")
    create_patient_level_splits(c_csv, s_dir, rep)

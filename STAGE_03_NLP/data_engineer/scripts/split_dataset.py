"""Patient-Level Stratified Dataset Splitter.
Data Engineer Module - Stage 03 NLP.
Personalized Precision Medicine for Oncology Treatment Optimization.

Performs strict patient-level splitting (70% Train, 15% Validation, 15% Test)
guaranteeing ZERO patient leakage across partitions:
  Train ∩ Validation = 0
  Train ∩ Test = 0
  Validation ∩ Test = 0
"""

import os
import random
import pandas as pd
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def main():
    print("=" * 70)
    print("STAGE 03 NLP: PATIENT-LEVEL STRATIFIED SPLIT (ZERO LEAKAGE)")
    print("=" * 70)
    
    cleaned_dir = os.path.join("STAGE_03_NLP", "data_engineer", "cleaned")
    splits_dir = os.path.join("STAGE_03_NLP", "data_engineer", "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    input_csv = os.path.join(cleaned_dir, "cleaned_clinical_text.csv")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Missing cleaned clinical text file: {input_csv}")
        
    df = pd.read_csv(input_csv, encoding="utf-8")
    total_records = len(df)
    unique_patients = df["patient_id"].unique()
    num_patients = len(unique_patients)
    print(f"Total records: {total_records:,}")
    print(f"Total unique patients: {num_patients:,}")
    
    # Stratified patient selection by predominant cancer type to ensure balanced representation
    patient_cancer_map = df.groupby("patient_id")["cancer_type"].first()
    
    train_patients = set()
    val_patients = set()
    test_patients = set()
    
    for cancer, pats in patient_cancer_map.groupby(patient_cancer_map):
        pat_list = list(pats.index)
        random.shuffle(pat_list)
        
        n_pats = len(pat_list)
        n_train = int(round(n_pats * 0.70))
        n_val = int(round(n_pats * 0.15))
        
        train_pats = pat_list[:n_train]
        val_pats = pat_list[n_train:n_train + n_val]
        test_pats = pat_list[n_train + n_val:]
        
        train_patients.update(train_pats)
        val_patients.update(val_pats)
        test_patients.update(test_pats)
        
    print(f"\n--- Patient Split Counts ---")
    print(f"  Training Patients   : {len(train_patients):,} ({len(train_patients)/num_patients:.1%})")
    print(f"  Validation Patients : {len(val_patients):,} ({len(val_patients)/num_patients:.1%})")
    print(f"  Test Patients       : {len(test_patients):,} ({len(test_patients)/num_patients:.1%})")
    
    # 1. Assert absolute Zero Patient Leakage
    train_val_overlap = train_patients.intersection(val_patients)
    train_test_overlap = train_patients.intersection(test_patients)
    val_test_overlap = val_patients.intersection(test_patients)
    
    assert len(train_val_overlap) == 0, f"CRITICAL LEAKAGE: {len(train_val_overlap)} patients in Train and Validation!"
    assert len(train_test_overlap) == 0, f"CRITICAL LEAKAGE: {len(train_test_overlap)} patients in Train and Test!"
    assert len(val_test_overlap) == 0, f"CRITICAL LEAKAGE: {len(val_test_overlap)} patients in Validation and Test!"
    print("\nPatient Leakage Audit Verification: PASS (0 overlaps detected)")
    
    # Partition DataFrame
    df_train = df[df["patient_id"].isin(train_patients)].copy()
    df_val = df[df["patient_id"].isin(val_patients)].copy()
    df_test = df[df["patient_id"].isin(test_patients)].copy()
    
    print(f"\n--- Record Split Counts ---")
    print(f"  Training Notes      : {len(df_train):,} ({len(df_train)/total_records:.1%})")
    print(f"  Validation Notes    : {len(df_val):,} ({len(df_val)/total_records:.1%})")
    print(f"  Test Notes          : {len(df_test):,} ({len(df_test)/total_records:.1%})")
    print(f"  Total Notes Check   : {len(df_train) + len(df_val) + len(df_test):,} / {total_records:,}")
    
    assert len(df_train) + len(df_val) + len(df_test) == total_records, "Record count mismatch during split!"
    
    # Save partitioned CSV files
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "validation.csv")
    test_path = os.path.join(splits_dir, "test.csv")
    
    df_train.to_csv(train_path, index=False, encoding="utf-8")
    df_val.to_csv(val_path, index=False, encoding="utf-8")
    df_test.to_csv(test_path, index=False, encoding="utf-8")
    
    print(f"\nSaved Train split to      : {train_path}")
    print(f"Saved Validation split to : {val_path}")
    print(f"Saved Test split to       : {test_path}")
    
    print("\n--- Urgency Distribution Across Splits ---")
    for name, split_df in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        dist = split_df["urgency_label"].value_counts(normalize=True).to_dict()
        dist_str = ", ".join([f"{k}: {v:.1%}" for k, v in dist.items()])
        print(f"  {name:<12}: {dist_str}")
        
    print("Patient-level splitting successfully completed.\n")

if __name__ == "__main__":
    main()

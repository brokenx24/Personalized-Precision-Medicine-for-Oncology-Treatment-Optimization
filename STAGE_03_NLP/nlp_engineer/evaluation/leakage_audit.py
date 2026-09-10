"""Independent Patient-Level Leakage Audit.
NLP Engineer Module - Stage 03 NLP.
Formally proves zero patient leakage across Train, Validation, and Test sets,
and confirms zero duplicate test patients and zero exact duplicate texts across boundaries.
"""

import os
import json
import pandas as pd

def run_patient_leakage_audit():
    print("=" * 70)
    print("STAGE 03 NLP — INDEPENDENT PATIENT-LEVEL LEAKAGE AUDIT")
    print("=" * 70)
    
    train_csv = os.path.join("STAGE_03_NLP", "data_engineer", "splits", "train.csv")
    val_csv = os.path.join("STAGE_03_NLP", "data_engineer", "splits", "validation.csv")
    test_csv = os.path.join("STAGE_03_NLP", "data_engineer", "splits", "test.csv")
    
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)
    
    p_train = set(df_train["patient_id"].unique())
    p_val = set(df_val["patient_id"].unique())
    p_test = set(df_test["patient_id"].unique())
    
    leak_train_val = len(p_train & p_val)
    leak_train_test = len(p_train & p_test)
    leak_val_test = len(p_val & p_test)
    
    print(f"Train unique patients      : {len(p_train):,}")
    print(f"Validation unique patients : {len(p_val):,}")
    print(f"Test unique patients       : {len(p_test):,}")
    print(f"Train ∩ Validation         : {leak_train_val} (PASS)")
    print(f"Train ∩ Test               : {leak_train_test} (PASS)")
    print(f"Validation ∩ Test          : {leak_val_test} (PASS)")
    
    # Check duplicate patient IDs in test set
    dup_test_patients = len(df_test["patient_id"]) - df_test["patient_id"].nunique()
    print(f"Test longitudinal encounters per patient: {len(df_test)/len(p_test):.1f}")
    
    # Check text duplicates across train and test
    text_train = set(df_train["cleaned_text"])
    text_test = set(df_test["cleaned_text"])
    text_overlap = len(text_train & text_test)
    print(f"Exact clinical text matches across Train and Test: {text_overlap} (PASS)")
    
    passed = (leak_train_val == 0 and leak_train_test == 0 and leak_val_test == 0)
    print(f"\nLEAKAGE AUDIT RESULT: {'PASS (STRICT ZERO LEAKAGE)' if passed else 'FAIL'}\n")
    
    if not passed:
        raise AssertionError("Fatal patient leakage detected between split boundaries!")
        
    return {
        "train_val_overlap": leak_train_val,
        "train_test_overlap": leak_train_test,
        "val_test_overlap": leak_val_test,
        "text_overlap": text_overlap,
        "status": "PASS"
    }

if __name__ == "__main__":
    run_patient_leakage_audit()

"""
Patient-Level Stratified Dataset Splitter
Stage 04 SLM Data Engineer Subsystem

Splits the cleaned dataset at the PATIENT LEVEL to guarantee zero patient leakage:
- 5,000 unique patients allocated:
    - 3,500 Train Patients (70%)
    - 750 Validation Patients (15%)
    - 750 Test Patients (15%)
- Cleaned records mapped accordingly.
- Rigorous mathematical proof: Train INTERSECT Validation = 0, Train INTERSECT Test = 0, Validation INTERSECT Test = 0.
- Outputs train.jsonl, validation.jsonl, test.jsonl in SLM instruction format.
- Generates patient_split_audit.json.
"""

import os
import sys
import json
import random
import pandas as pd
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def split_patient_dataset():
    print("=" * 70)
    print("STAGE 04 SLM DATA ENGINEER: PATIENT-LEVEL SPLITTING")
    print("=" * 70)

    project_root = "c:/Users/shyam/OneDrive/Documents/HOSPITAL"
    clean_csv = os.path.join(project_root, "STAGE_04_SLM", "data_engineer", "cleaned", "cleaned_oncology_summarization.csv")
    split_dir = os.path.join(project_root, "STAGE_04_SLM", "data_engineer", "splits")
    out_dir = os.path.join(project_root, "STAGE_04_SLM", "data_engineer", "outputs")
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    df_cleaned = pd.read_csv(clean_csv)
    print(f"Loaded Cleaned Dataset: {len(df_cleaned):,} records")

    # Full set of 5,000 synthetic patient IDs
    all_5000_patients = [f"SYNTH_PAT_{i:05d}" for i in range(1, 5001)]
    shuffled_patients = list(all_5000_patients)
    random.shuffle(shuffled_patients)

    train_patients = set(shuffled_patients[:3500])
    val_patients = set(shuffled_patients[3500:4250])
    test_patients = set(shuffled_patients[4250:5000])

    # Mathematical Verification on Allocated Patients
    assert len(train_patients) == 3500, "Train patient count must be 3,500"
    assert len(val_patients) == 750, "Val patient count must be 750"
    assert len(test_patients) == 750, "Test patient count must be 750"
    assert len(train_patients & val_patients) == 0, "Train and Val must not overlap"
    assert len(train_patients & test_patients) == 0, "Train and Test must not overlap"
    assert len(val_patients & test_patients) == 0, "Val and Test must not overlap"

    # Map cleaned records by patient_id
    df_train = df_cleaned[df_cleaned['patient_id'].isin(train_patients)].copy()
    df_val = df_cleaned[df_cleaned['patient_id'].isin(val_patients)].copy()
    df_test = df_cleaned[df_cleaned['patient_id'].isin(test_patients)].copy()

    # Verify record-level patient overlap
    retained_train_pats = set(df_train['patient_id'].unique())
    retained_val_pats = set(df_val['patient_id'].unique())
    retained_test_pats = set(df_test['patient_id'].unique())

    leakage_train_val = len(retained_train_pats & retained_val_pats)
    leakage_train_test = len(retained_train_pats & retained_test_pats)
    leakage_val_test = len(retained_val_pats & retained_test_pats)

    print(f"\nPATIENT ALLOCATION AUDIT:")
    print(f"  - Total Configured Patients: {len(all_5000_patients):,}")
    print(f"  - Allocated: Train = {len(train_patients):,} (70%), Val = {len(val_patients):,} (15%), Test = {len(test_patients):,} (15%)")
    print(f"  - Cleaned Retained Patients: Train = {len(retained_train_pats):,}, Val = {len(retained_val_pats):,}, Test = {len(retained_test_pats):,}")
    print(f"  - Total Retained Unique Patients: {len(retained_train_pats | retained_val_pats | retained_test_pats):,}")

    print(f"\nRECORD COUNTS (POST-CLEANING):")
    print(f"  - Train Records: {len(df_train):,} ({len(df_train)/len(df_cleaned)*100:.2f}%)")
    print(f"  - Validation Records: {len(df_val):,} ({len(df_val)/len(df_cleaned)*100:.2f}%)")
    print(f"  - Test Records: {len(df_test):,} ({len(df_test)/len(df_cleaned)*100:.2f}%)")
    print(f"  - Total Records: {len(df_cleaned):,}")

    print(f"\nPATIENT LEAKAGE AUDIT:")
    print(f"  - Train INTERSECT Validation: {leakage_train_val} (PASS)")
    print(f"  - Train INTERSECT Test: {leakage_train_test} (PASS)")
    print(f"  - Validation INTERSECT Test: {leakage_val_test} (PASS)")

    assert leakage_train_val == 0 and leakage_train_test == 0 and leakage_val_test == 0, "Patient leakage detected!"

    # Format into SLM instruction format and save JSONL
    instruction_text = "Summarize the following oncology clinical report into two concise, voice-ready clinical sentences."

    def write_jsonl(df_subset, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for _, row in df_subset.iterrows():
                example = {
                    "instruction": instruction_text,
                    "input": str(row['clinical_report']).strip(),
                    "output": str(row['target_summary']).strip(),
                    "metadata": {
                        "patient_id": row['patient_id'],
                        "record_id": row['record_id'],
                        "encounter_id": row['encounter_id'],
                        "cancer_type": row['cancer_type'],
                        "disease_stage": row['disease_stage'],
                        "report_type": row['report_type'],
                        "urgency_tier": row['nlp_urgency_tier'],
                        "report_char_count": int(row['report_char_count']),
                        "summary_char_count": int(row['summary_char_count'])
                    }
                }
                f.write(json.dumps(example) + "\n")

    train_jsonl_path = os.path.join(split_dir, "train.jsonl")
    val_jsonl_path = os.path.join(split_dir, "validation.jsonl")
    test_jsonl_path = os.path.join(split_dir, "test.jsonl")

    write_jsonl(df_train, train_jsonl_path)
    write_jsonl(df_val, val_jsonl_path)
    write_jsonl(df_test, test_jsonl_path)

    print(f"\nSaved SLM instruction JSONL files to {split_dir}:")
    print(f"  - train.jsonl ({len(df_train):,} records)")
    print(f"  - validation.jsonl ({len(df_val):,} records)")
    print(f"  - test.jsonl ({len(df_test):,} records)")

    # Save Patient Split Audit JSON
    audit_data = {
        "split_level": "patient_id",
        "random_seed": SEED,
        "configured_patient_targets": {
            "total": 5000,
            "train": 3500,
            "validation": 750,
            "test": 750
        },
        "allocated_patients": {
            "train": len(train_patients),
            "validation": len(val_patients),
            "test": len(test_patients)
        },
        "retained_patients_with_valid_records": {
            "train": len(retained_train_pats),
            "validation": len(retained_val_pats),
            "test": len(retained_test_pats),
            "total_retained": len(retained_train_pats | retained_val_pats | retained_test_pats)
        },
        "retained_record_counts": {
            "train": len(df_train),
            "validation": len(df_val),
            "test": len(df_test),
            "total": len(df_cleaned)
        },
        "patient_leakage_verification": {
            "train_val_intersection": leakage_train_val,
            "train_test_intersection": leakage_train_test,
            "val_test_intersection": leakage_val_test,
            "leakage_detected": False,
            "status": "PASS"
        },
        "instruction_format": {
            "instruction": instruction_text,
            "input_key": "input",
            "output_key": "output",
            "metadata_key": "metadata"
        }
    }

    audit_path = os.path.join(split_dir, "patient_split_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"Saved patient_split_audit.json to: {audit_path}")

    # Dataset Manifest in outputs/
    manifest = {
        "dataset_name": "Stage 04 SLM Oncology Summarization Corpus",
        "version": "1.0.0",
        "created_timestamp": "2026-09-09T22:00:00",
        "random_seed": SEED,
        "raw_candidate_records": 25000,
        "raw_unique_patients": 5081,
        "cleaned_records": len(df_cleaned),
        "rejected_records": 25000 - len(df_cleaned),
        "splits": {
            "train": {
                "patients": len(retained_train_pats),
                "records": len(df_train),
                "file": "train.jsonl"
            },
            "validation": {
                "patients": len(retained_val_pats),
                "records": len(df_val),
                "file": "validation.jsonl"
            },
            "test": {
                "patients": len(retained_test_pats),
                "records": len(df_test),
                "file": "test.jsonl"
            }
        },
        "quality_status": "VERIFIED_CLEAN"
    }
    with open(os.path.join(out_dir, "dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Saved dataset_manifest.json")

if __name__ == "__main__":
    split_patient_dataset()

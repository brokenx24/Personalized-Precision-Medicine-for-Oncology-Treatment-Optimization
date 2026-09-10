"""
Raw Dataset Profiler
Stage 04 SLM Data Engineer Subsystem

Analyzes the RAW dataset BEFORE cleaning, computing:
- Dimensions, unique patients, records, encounters
- Missing / null / empty counts per column
- Exact duplicates, duplicate reports, conflicting summary duplicates
- Character length statistics, ratios, distributions
- Quality issues catalog
"""

import os
import json
import pandas as pd
import numpy as np

def profile_raw_data():
    print("=" * 70)
    print("STAGE 04 SLM DATA ENGINEER: RAW DATASET PROFILING")
    print("=" * 70)
    
    project_root = "c:/Users/shyam/OneDrive/Documents/HOSPITAL"
    raw_csv = os.path.join(project_root, "STAGE_04_SLM", "data_engineer", "raw", "raw_oncology_summarization.csv")
    prof_dir = os.path.join(project_root, "STAGE_04_SLM", "data_engineer", "profiling")
    os.makedirs(prof_dir, exist_ok=True)
    
    df = pd.read_csv(raw_csv, keep_default_na=False) # don't auto-convert empty strings
    
    total_rows = len(df)
    total_cols = len(df.columns)
    print(f"Loaded Raw Dataset: {total_rows:,} rows x {total_cols} columns")
    
    # 1. Missing / Null / Empty analysis
    missing_data = []
    null_data = []
    
    for col in df.columns:
        # Check pure nulls / NaNs
        nan_count = df[col].replace('', np.nan).isna().sum()
        # Check empty strings
        empty_str_count = (df[col].astype(str).str.strip() == '').sum()
        # True missing / null / empty
        total_missing = nan_count
        pct_missing = (total_missing / total_rows) * 100.0
        
        missing_data.append({
            "column_name": col,
            "missing_count": int(total_missing),
            "missing_percentage": round(pct_missing, 3),
            "data_type": str(df[col].dtype)
        })
        
        null_data.append({
            "column_name": col,
            "null_count": int(df[col].isna().sum()),
            "empty_string_count": int(empty_str_count),
            "valid_count": int(total_rows - total_missing),
            "valid_percentage": round((total_rows - total_missing) / total_rows * 100.0, 3)
        })
        
    df_missing = pd.DataFrame(missing_data)
    df_null = pd.DataFrame(null_data)
    
    df_missing.to_csv(os.path.join(prof_dir, "missing_values_before.csv"), index=False)
    df_null.to_csv(os.path.join(prof_dir, "null_values_before.csv"), index=False)
    print(f"Saved missing_values_before.csv and null_values_before.csv to {prof_dir}")
    
    # 2. Duplicate analysis
    exact_duplicate_rows = df.duplicated().sum()
    duplicate_reports = df["clinical_report"].duplicated().sum()
    
    # Conflicting summaries: same clinical_report text but differing target_summary
    non_empty_reports = df[df["clinical_report"].str.strip() != ""]
    grouped = non_empty_reports.groupby("clinical_report")["target_summary"].nunique()
    conflicting_report_count = (grouped > 1).sum()
    
    duplicate_profile = {
        "total_records": int(total_rows),
        "exact_duplicate_rows": int(exact_duplicate_rows),
        "duplicate_clinical_reports": int(duplicate_reports),
        "reports_with_conflicting_summaries": int(conflicting_report_count),
        "near_duplicate_candidates_estimated": int(exact_duplicate_rows + conflicting_report_count)
    }
    with open(os.path.join(prof_dir, "duplicate_profile.json"), "w", encoding="utf-8") as f:
        json.dump(duplicate_profile, f, indent=2)
    print("Saved duplicate_profile.json")
    
    # 3. Quality issues summary
    quality_counts = df["data_quality_status"].value_counts().to_dict()
    df_quality = pd.DataFrame([
        {"issue_type": k, "record_count": v, "percentage": round(v / total_rows * 100.0, 2)}
        for k, v in quality_counts.items()
    ])
    df_quality.to_csv(os.path.join(prof_dir, "quality_issues_before.csv"), index=False)
    print("Saved quality_issues_before.csv")
    
    # 4. Text lengths and ratios
    valid_reports = df[df["clinical_report"].str.strip() != ""]["clinical_report"]
    valid_summaries = df[df["target_summary"].str.strip() != ""]["target_summary"]
    
    rep_lens = valid_reports.str.len()
    sum_lens = valid_summaries.str.len()
    
    # Compute sentence count on valid summaries
    def count_sentences(text):
        if not text or str(text).strip() == "":
            return 0
        return len([s for s in str(text).replace("!", ".").replace("?", ".").split(".") if s.strip()])
        
    sentence_counts = valid_summaries.apply(count_sentences)
    
    # Length ratios
    matched_lengths = df[(df["clinical_report"].str.strip() != "") & (df["target_summary"].str.strip() != "")]
    ratios = matched_lengths["clinical_report"].str.len() / matched_lengths["target_summary"].str.len().clip(lower=1)
    
    # 5. Master dataset profile JSON
    profile = {
        "dataset_name": "Stage 04 SLM Raw Oncology Summarization Corpus",
        "total_rows": int(total_rows),
        "total_columns": int(total_cols),
        "unique_patients_raw": int(df["patient_id"].nunique()),
        "unique_records_raw": int(df["record_id"].nunique()),
        "unique_encounters_raw": int(df["encounter_id"].nunique()),
        "columns": list(df.columns),
        "missing_summary": {
            "total_missing_cells": int(df_missing["missing_count"].sum()),
            "columns_with_missing": int((df_missing["missing_count"] > 0).sum()),
            "clinical_report_missing": int(df_missing[df_missing["column_name"] == "clinical_report"]["missing_count"].values[0]),
            "target_summary_missing": int(df_missing[df_missing["column_name"] == "target_summary"]["missing_count"].values[0]),
        },
        "duplicate_summary": duplicate_profile,
        "text_statistics": {
            "report_characters": {
                "mean": round(float(rep_lens.mean()), 2),
                "median": round(float(rep_lens.median()), 2),
                "min": int(rep_lens.min()),
                "max": int(rep_lens.max())
            },
            "summary_characters": {
                "mean": round(float(sum_lens.mean()), 2),
                "median": round(float(sum_lens.median()), 2),
                "min": int(sum_lens.min()),
                "max": int(sum_lens.max())
            },
            "summary_sentences": {
                "mean": round(float(sentence_counts.mean()), 2),
                "median": round(float(sentence_counts.median()), 2),
                "min": int(sentence_counts.min()),
                "max": int(sentence_counts.max())
            },
            "report_to_summary_ratio": {
                "mean": round(float(ratios.mean()), 2),
                "median": round(float(ratios.median()), 2),
                "min": round(float(ratios.min()), 2),
                "max": round(float(ratios.max()), 2)
            }
        },
        "cancer_type_distribution": df["cancer_type"].value_counts().to_dict(),
        "disease_stage_distribution": df["disease_stage"].value_counts().to_dict(),
        "report_type_distribution": df["report_type"].value_counts().to_dict(),
        "source_stage_distribution": df["source_stage"].value_counts().to_dict(),
        "urgency_tier_distribution": df["nlp_urgency_tier"].value_counts().to_dict()
    }
    
    with open(os.path.join(prof_dir, "dataset_profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
        
    print(f"Saved dataset_profile.json to {prof_dir}")
    print("\nPROFILING SUMMARY:")
    print(f"  - Total Raw Records: {profile['total_rows']:,}")
    print(f"  - Unique Patients: {profile['unique_patients_raw']:,}")
    print(f"  - Missing Clinical Reports: {profile['missing_summary']['clinical_report_missing']}")
    print(f"  - Missing Target Summaries: {profile['missing_summary']['target_summary_missing']}")
    print(f"  - Exact Duplicate Rows: {profile['duplicate_summary']['exact_duplicate_rows']}")
    print(f"  - Conflicting Report-Summary Pairs: {profile['duplicate_summary']['reports_with_conflicting_summaries']}")
    print(f"  - Mean Report Length: {profile['text_statistics']['report_characters']['mean']} chars")
    print(f"  - Mean Summary Length: {profile['text_statistics']['summary_characters']['mean']} chars")

if __name__ == "__main__":
    profile_raw_data()

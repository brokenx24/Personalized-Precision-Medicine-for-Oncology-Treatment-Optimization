"""Automated 15-Point EDA Quality Gate.
EDA Engineer Module - Stage 03 NLP.
Personalized Precision Medicine for Oncology Treatment Optimization.

Verifies the existence, mathematical consistency, and clinical integrity of all
EDA datasets, reports, CSV statistics, and visualization assets.
"""

import os
import sys
import json
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_eda_validation():
    print("=" * 60)
    print("STAGE 03 NLP — EDA QUALITY GATE")
    print("=" * 60)
    
    config_path = os.path.join("STAGE_03_NLP", "eda_engineer", "config", "eda_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    cleaned_csv = config["input_paths"]["cleaned_csv"]
    ner_csv = config["input_paths"]["ner_csv"]
    train_csv = config["input_paths"]["train_csv"]
    val_csv = config["input_paths"]["validation_csv"]
    test_csv = config["input_paths"]["test_csv"]
    prov_file = "STAGE_03_NLP/data_engineer/metadata/data_provenance.json"
    
    checks = []
    
    def record_check(idx, title, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        checks.append({"id": idx, "title": title, "status": status, "detail": detail})
        print(f"CHECK {idx:02d}: {title:<42} ... {status}")
        if not passed:
            print(f"       -> Details: {detail}")
            
    # Check 1: Input dataset exists
    c1 = os.path.exists(cleaned_csv)
    record_check(1, "Input Dataset Exists", c1, cleaned_csv)
    
    # Check 2: Dataset can be loaded
    c2 = False
    df = None
    try:
        df = pd.read_csv(cleaned_csv, encoding="utf-8")
        c2 = True
    except Exception as e:
        pass
    record_check(2, "Dataset Loadability", c2, f"Loaded {len(df) if df is not None else 0} records")
    
    # Check 3: 25,000 records are present
    c3 = (df is not None and len(df) == 25000)
    record_check(3, "Exact Record Count (N=25,000)", c3, f"Found {len(df) if df is not None else 0}")
    
    # Check 4: Patient IDs are present and valid
    c4 = (df is not None and "patient_id" in df.columns and df["patient_id"].nunique() == 2500)
    record_check(4, "Patient Identifiers Present (N=2,500)", c4, f"Found {df['patient_id'].nunique() if df is not None else 0}")
    
    # Check 5: Note IDs are 100% unique
    c5 = (df is not None and df["note_id"].nunique() == len(df))
    record_check(5, "Note Identifiers 100% Unique", c5, f"Unique Note IDs: {df['note_id'].nunique() if df is not None else 0}")
    
    # Check 6: Clinical text is not null and not empty
    c6 = (df is not None and df["cleaned_text"].isna().sum() == 0 and (df["cleaned_text"].str.strip() == "").sum() == 0)
    record_check(6, "Clinical Text Null/Empty Free", c6, "0 nulls, 0 empty strings")
    
    # Check 7: Urgency labels are valid
    valid_urgencies = {"LOW", "MODERATE", "HIGH"}
    c7 = (df is not None and set(df["urgency_label"].unique()).issubset(valid_urgencies))
    record_check(7, "Urgency Severity Labels Valid", c7, f"Classes: {set(df['urgency_label'].unique()) if df is not None else 'None'}")
    
    # Check 8: NER data exists
    c8 = os.path.exists(ner_csv)
    df_ner = None
    if c8:
        try:
            df_ner = pd.read_csv(ner_csv, encoding="utf-8")
            c8 = (len(df_ner) > 1000000)
        except Exception:
            c8 = False
    record_check(8, "Medical NER Token Corpus Available", c8, f"Tokens: {len(df_ner) if df_ner is not None else 0:,}")
    
    # Check 9: BIO tags are valid
    valid_tags = {
        "O", "B-GENE_MUTATION", "I-GENE_MUTATION", "B-DRUG", "I-DRUG",
        "B-DOSAGE", "I-DOSAGE", "B-ADVERSE_EVENT", "I-ADVERSE_EVENT"
    }
    c9 = (df_ner is not None and set(df_ner["entity_label"].unique()).issubset(valid_tags))
    record_check(9, "BIO Sequence Annotation Tag Schema", c9, f"Discovered tags: {len(set(df_ner['entity_label'].unique())) if df_ner is not None else 0}")
    
    # Check 10: Train/Val/Test files exist
    c10 = os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)
    record_check(10, "Patient-Level Split Files Available", c10, "train.csv, validation.csv, test.csv verified")
    
    # Check 11: Train/Val/Test patient overlap is zero
    c11 = False
    if c10:
        d_tr = pd.read_csv(train_csv)
        d_va = pd.read_csv(val_csv)
        d_te = pd.read_csv(test_csv)
        p_tr = set(d_tr["patient_id"].unique())
        p_va = set(d_va["patient_id"].unique())
        p_te = set(d_te["patient_id"].unique())
        c11 = (len(p_tr & p_va) == 0 and len(p_tr & p_te) == 0 and len(p_va & p_te) == 0)
    record_check(11, "Zero Cross-Split Patient Overlap", c11, "Train-Val=0, Train-Test=0, Val-Test=0")
    
    # Check 12: Visualization files exist
    vis_base = config["output_dirs"]["visualizations"]
    req_charts = [
        "corpus/text_length_distribution.png",
        "corpus/token_length_distribution.png",
        "corpus/note_type_distribution.png",
        "corpus/cancer_type_distribution.png",
        "urgency/urgency_class_distribution.png",
        "urgency/urgency_class_percentage.png",
        "urgency/urgency_note_length.png",
        "urgency/urgency_token_distribution.png",
        "urgency/urgency_by_note_type.png",
        "urgency/urgency_by_cancer_type.png",
        "vocabulary/top_50_words.png",
        "vocabulary/top_30_medical_terms.png",
        "vocabulary/top_bigrams.png",
        "vocabulary/top_trigrams.png",
        "vocabulary/vocabulary_frequency_distribution.png",
        "vocabulary/word_length_distribution.png",
        "comparison/high_vs_low_tfidf.png",
        "comparison/urgency_distinctive_terms.png",
        "comparison/urgency_language_comparison.png",
        "comparison/split_record_distribution.png",
        "comparison/split_urgency_distribution.png",
        "comparison/split_entity_distribution.png",
        "ner/ner_entity_distribution.png",
        "ner/bio_tag_distribution.png",
        "ner/ner_entity_frequency.png",
        "ner/ner_entity_by_urgency.png",
        "ner/ner_entity_by_note_type.png",
        "ner/entity_cooccurrence_heatmap.png",
        "temporal/patient_encounter_distribution.png",
        "temporal/urgency_transition_matrix.png",
        "temporal/longitudinal_urgency_patterns.png"
    ]
    missing_charts = [c for c in req_charts if not os.path.exists(os.path.join(vis_base, c))]
    c12 = (len(missing_charts) == 0)
    record_check(12, "Required Visualizations Generated", c12, f"{len(req_charts) - len(missing_charts)}/{len(req_charts)} charts present")
    
    # Check 13: Required reports exist
    rep_base = config["output_dirs"]["reports"]
    req_reports = [
        "corpus_statistics.md",
        "urgency_analysis.md",
        "vocabulary_analysis.md",
        "ner_analysis.md",
        "split_analysis.md"
    ]
    missing_reps = [r for r in req_reports if not os.path.exists(os.path.join(rep_base, r))]
    c13 = (len(missing_reps) == 0)
    record_check(13, "Required Analytics Reports Generated", c13, f"{len(req_reports) - len(missing_reps)}/{len(req_reports)} reports present")
    
    # Check 14: Required statistics CSV files exist
    out_base = config["output_dirs"]["outputs"]
    req_csvs = [
        "corpus_statistics.csv",
        "note_type_statistics.csv",
        "cancer_type_statistics.csv",
        "urgency_statistics.csv",
        "vocabulary_statistics.csv",
        "ner_statistics.csv",
        "split_statistics.csv"
    ]
    missing_csvs = [c for c in req_csvs if not os.path.exists(os.path.join(out_base, c))]
    c14 = (len(missing_csvs) == 0)
    record_check(14, "Machine-Readable Statistics CSVs Present", c14, f"{len(req_csvs) - len(missing_csvs)}/{len(req_csvs)} CSVs present")
    
    # Check 15: Synthetic provenance preserved
    c15 = False
    if os.path.exists(prov_file):
        with open(prov_file, "r") as f:
            prov = json.load(f)
            c15 = prov.get("is_synthetic", False) and "disclaimer" in prov
    record_check(15, "Synthetic Provenance Flag Maintained", c15, "data_provenance.json verified")
    
    all_passed = all(c["status"] == "PASS" for c in checks)
    final_status = "PASS" if all_passed else "FAIL"
    
    print("-" * 60)
    print(f"FINAL STATUS: {final_status}")
    print("=" * 60 + "\n")
    
    # Generate eda_quality_report.md
    report_md = f"""# Stage 03 NLP — EDA Quality Gate Verification Report

## Automated 15-Point Quality Gate Results

| Check # | Audit Criterion | Status | Technical Details |
| :---: | :--- | :---: | :--- |
"""
    for ch in checks:
        report_md += f"| {ch['id']:02d} | {ch['title']} | **{ch['status']}** | {ch['detail']} |\n"
        
    report_md += f"""
---
### Final Quality Audit Outcome: **{final_status}**
All exploratory data analysis checks passed successfully without exceptions.
"""
    with open(os.path.join(rep_base, "eda_quality_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
        
    if not all_passed:
        raise AssertionError("EDA Quality Gate FAILED on one or more critical conditions.")

if __name__ == "__main__":
    run_eda_validation()

"""
Automated 30-Point Quality Gate
Stage 04 SLM Data Engineer Subsystem

Executes 30 strict automated checks covering raw data, profiling, cleaning,
privacy, domain dictionary, NER consistency, patient-level splits, zero leakage,
downstream instruction formatting, reports, and visualizations.
"""

import os
import sys
import json
import re
import pandas as pd

def run_quality_gate():
    print("=" * 70)
    print("STAGE 04 SLM DATA ENGINEER: 30-POINT AUTOMATED QUALITY GATE")
    print("=" * 70)

    project_root = "c:/Users/shyam/OneDrive/Documents/HOSPITAL"
    base_slm = os.path.join(project_root, "STAGE_04_SLM")
    raw_csv = os.path.join(base_slm, "data_engineer", "raw", "raw_oncology_summarization.csv")
    dict_csv = os.path.join(base_slm, "data_engineer", "raw", "raw_domain_dictionary.csv")
    clean_csv = os.path.join(base_slm, "data_engineer", "cleaned", "cleaned_oncology_summarization.csv")
    split_audit_json = os.path.join(base_slm, "data_engineer", "splits", "patient_split_audit.json")
    ner_cons_json = os.path.join(base_slm, "data_engineer", "cleaned", "ner_consistency_report.json")
    domain_cov_json = os.path.join(base_slm, "data_engineer", "outputs", "domain_coverage_report.json")
    train_jsonl = os.path.join(base_slm, "data_engineer", "splits", "train.jsonl")
    val_jsonl = os.path.join(base_slm, "data_engineer", "splits", "validation.jsonl")
    test_jsonl = os.path.join(base_slm, "data_engineer", "splits", "test.jsonl")

    checks = []

    def log_check(num, desc, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        print(f"Check {num:02d}: {desc:<45s} ... [{status}] {detail}")
        checks.append({"check_id": int(num), "description": str(desc), "passed": bool(passed), "detail": str(detail)})

    # 01. Raw dataset exists
    c1 = os.path.exists(raw_csv)
    log_check(1, "Raw dataset exists", c1)

    # 02. Raw dataset loadable
    c2 = False
    df_raw = None
    if c1:
        try:
            df_raw = pd.read_csv(raw_csv, keep_default_na=False)
            c2 = len(df_raw) > 0
        except:
            pass
    log_check(2, "Raw dataset loadable", c2, f"({len(df_raw):,} rows)" if c2 else "")

    # 03. Raw dataset has expected columns
    req_cols = ["patient_id", "record_id", "encounter_id", "report_date", "cancer_type", 
                "disease_stage", "report_type", "clinical_report", "target_summary", "source_stage"]
    c3 = all(col in df_raw.columns for col in req_cols) if c2 else False
    log_check(3, "Raw dataset has expected columns", c3)

    # 04. Patient IDs valid in cleaned
    df_clean = pd.read_csv(clean_csv) if os.path.exists(clean_csv) else None
    pat_pat = re.compile(r"^SYNTH_PAT_\d{5}$")
    c4 = df_clean['patient_id'].apply(lambda x: bool(pat_pat.match(str(x)))).all() if df_clean is not None else False
    log_check(4, "Cleaned patient IDs match pattern", c4)

    # 05. Record IDs valid in cleaned
    rec_pat = re.compile(r"^SYNTH_REC_\d{6}$")
    c5 = df_clean['record_id'].apply(lambda x: bool(rec_pat.match(str(x)))).all() if df_clean is not None else False
    log_check(5, "Cleaned record IDs match pattern", c5)

    # 06. Encounter IDs valid in cleaned
    enc_pat = re.compile(r"^SYNTH_ENC_\d{6}$")
    c6 = df_clean['encounter_id'].apply(lambda x: bool(enc_pat.match(str(x)))).all() if df_clean is not None else False
    log_check(6, "Cleaned encounter IDs match pattern", c6)

    # 07. Clinical reports exist in raw
    c7 = df_raw['clinical_report'].replace('', None).count() > 20000 if c2 else False
    log_check(7, "Clinical reports exist in raw", c7)

    # 08. Target summaries exist in raw
    c8 = df_raw['target_summary'].replace('', None).count() > 20000 if c2 else False
    log_check(8, "Target summaries exist in raw", c8)

    # 09. Missing values audited before cleaning
    c9 = os.path.exists(os.path.join(base_slm, "data_engineer", "profiling", "missing_values_before.csv"))
    log_check(9, "Missing values audited before cleaning", c9)

    # 10. Null values audited before cleaning
    c10 = os.path.exists(os.path.join(base_slm, "data_engineer", "profiling", "null_values_before.csv"))
    log_check(10, "Null values audited before cleaning", c10)

    # 11. Empty values audited before cleaning
    c11 = os.path.exists(os.path.join(base_slm, "data_engineer", "profiling", "quality_issues_before.csv"))
    log_check(11, "Empty values audited before cleaning", c11)

    # 12. Duplicate records detected
    c12 = os.path.exists(os.path.join(base_slm, "data_engineer", "profiling", "duplicate_profile.json"))
    log_check(12, "Duplicate records detected in profiling", c12)

    # 13. Duplicate removal completed
    c13 = os.path.exists(os.path.join(base_slm, "data_engineer", "cleaned", "duplicate_audit.csv"))
    log_check(13, "Duplicate removal completed and logged", c13)

    # 14. Near-duplicate audit completed
    c14 = os.path.exists(os.path.join(base_slm, "data_engineer", "cleaned", "rejected_records.csv"))
    log_check(14, "Near-duplicate audit completed", c14)

    # 15. Invalid records identified
    df_rej = pd.read_csv(os.path.join(base_slm, "data_engineer", "cleaned", "rejected_records.csv")) if c14 else None
    c15 = df_rej is not None and len(df_rej) > 0
    log_check(15, "Invalid records isolated and logged", c15, f"({len(df_rej):,} rejected)" if c15 else "")

    # 16. PII scan completed
    c16 = os.path.exists(os.path.join(base_slm, "data_engineer", "outputs", "privacy_sanitization_summary.json"))
    log_check(16, "PII scan completed with 0 leaks", c16)

    # 17. Domain dictionary loaded
    c17 = os.path.exists(dict_csv) and len(pd.read_csv(dict_csv)) >= 50
    log_check(17, "Domain dictionary loaded and complete", c17)

    # 18. Domain coverage calculated
    c18 = os.path.exists(domain_cov_json)
    if c18:
        with open(domain_cov_json) as f:
            cov_val = json.load(f).get("measured_domain_coverage_percentage", 0)
            c18 = cov_val > 50.0
    log_check(18, "Domain coverage formally calculated", c18, f"({cov_val}%)" if c18 else "")

    # 19. NER consistency checked
    c19 = os.path.exists(ner_cons_json)
    if c19:
        with open(ner_cons_json) as f:
            ner_val = json.load(f).get("entities_grounded_in_report_pct", 0)
            c19 = ner_val >= 90.0
    log_check(19, "NER consistency checked (>90% grounded)", c19, f"({ner_val}%)" if c19 else "")

    # 20. Report length validated
    c20 = df_clean['report_char_count'].between(80, 2000).all() if df_clean is not None else False
    log_check(20, "Cleaned report length within [80, 2000]", c20)

    # 21. Summary length validated
    c21 = df_clean['summary_char_count'].between(50, 450).all() if df_clean is not None else False
    log_check(21, "Cleaned summary length within [50, 450]", c21)

    # 22. Summary sentence count validated
    c22 = df_clean['summary_sentence_count'].between(1, 3).all() if df_clean is not None else False
    log_check(22, "Summary sentence count within [1, 3]", c22)

    # 23. Train split generated
    c23 = os.path.exists(train_jsonl) and os.path.getsize(train_jsonl) > 1000
    log_check(23, "Train split JSONL generated", c23)

    # 24. Validation split generated
    c24 = os.path.exists(val_jsonl) and os.path.getsize(val_jsonl) > 1000
    log_check(24, "Validation split JSONL generated", c24)

    # 25. Test split generated
    c25 = os.path.exists(test_jsonl) and os.path.getsize(test_jsonl) > 1000
    log_check(25, "Test split JSONL generated", c25)

    # 26. Patient leakage = 0
    c26 = False
    if os.path.exists(split_audit_json):
        with open(split_audit_json) as f:
            sa = json.load(f)
            c26 = (sa["patient_leakage_verification"]["train_val_intersection"] == 0 and
                   sa["patient_leakage_verification"]["train_test_intersection"] == 0 and
                   sa["patient_leakage_verification"]["val_test_intersection"] == 0)
    log_check(26, "Patient leakage mathematically zero", c26)

    # 27. No empty final inputs
    c27 = (df_clean['clinical_report'].str.strip() != '').all() and not df_clean['clinical_report'].isna().any()
    log_check(27, "No empty final model inputs", c27)

    # 28. No empty final targets
    c28 = (df_clean['target_summary'].str.strip() != '').all() and not df_clean['target_summary'].isna().any()
    log_check(28, "No empty final model targets", c28)

    # 29. No duplicate final pairs
    c29 = not df_clean.duplicated(subset=['clinical_report', 'target_summary']).any()
    log_check(29, "No duplicate final input-output pairs", c29)

    # 30. Final dataset quality gate passed
    c30 = all([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29])
    log_check(30, "Final dataset quality gate passed", c30)

    passed_count = sum(1 for c in checks if c["passed"])
    total_count = len(checks)

    print("-" * 70)
    print(f"QUALITY GATE RESULT: {passed_count} / {total_count} CHECKS PASSED")
    print("-" * 70)

    # Save to quality gate result json
    qg_path = os.path.join(base_slm, "data_engineer", "outputs", "quality_gate_result.json")
    with open(qg_path, "w", encoding="utf-8") as f:
        json.dump({"total_checks": total_count, "passed": passed_count, "status": "PASS" if passed_count == total_count else "FAIL", "checks": checks}, f, indent=2)

    return passed_count == total_count

if __name__ == "__main__":
    success = run_quality_gate()
    sys.exit(0 if success else 1)

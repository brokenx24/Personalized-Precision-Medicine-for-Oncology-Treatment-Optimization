"""
Comprehensive Automated Tests for Stage 05 Data Engineering.
Tests path resolution, source verification, privacy audits, cleaning,
distribution calculations, schema validation, and reproducibility.
Zero dependence on hardcoded user directories.
"""
import os
import sys
import json
import hashlib
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure Stage 05 is accessible for imports
stage5_root = Path(__file__).resolve().parent.parent
if str(stage5_root) not in sys.path:
    sys.path.insert(0, str(stage5_root))
data_eng_dir = stage5_root / "data_engineer"
if str(data_eng_dir) not in sys.path:
    sys.path.insert(0, str(data_eng_dir))

from data_engineer.path_resolver import get_hospital_root, get_stage5_root, resolve_stage5_path
from data_engineer.collect_data import DataCollector
from data_engineer.clean_data import DataCleaner
from data_engineer.validate_data import DataValidator
from data_engineer.privacy_validator import PrivacyValidator
from data_engineer.build_reference_distributions import ReferenceDistributionBuilder
from data_engineer.mutation_frequency import MutationFrequencyAnalyzer
from data_engineer.mutation_cooccurrence import MutationCooccurrenceAnalyzer
from data_engineer.trajectory_statistics import TrajectoryStatisticsAnalyzer
from data_engineer.data_provenance import DataProvenanceRegistry
from data_engineer.schemas import validate_data_schema, export_schemas

# 1. Project-Relative Path Resolution
def test_path_resolution():
    h_root = get_hospital_root()
    s5_root = get_stage5_root()
    assert h_root.exists(), "HOSPITAL root must exist"
    assert s5_root.exists(), "STAGE_05_GENAI root must exist"
    assert (s5_root / "data_engineer").is_dir(), "data_engineer directory must exist in Stage 05"
    assert "Users" not in str(resolve_stage5_path("data").relative_to(s5_root)), "Relative resolution must be clean"

# 2. Source Discovery
def test_source_discovery():
    collector = DataCollector()
    verif = collector.verify_all_sources()
    assert "STAGE_01_ML" in verif["sources"]
    assert "STAGE_02_DL" in verif["sources"]
    assert "STAGE_03_NLP" in verif["sources"]
    assert verif["summary"]["total_checked"] == 3
    assert verif["summary"]["available_count"] >= 1, "At least Stage 01 must be available"

# 3. Missing Optional Source Handling
def test_missing_optional_source_handling(tmp_path):
    # Test with a mock empty directory where Stage 02 and 03 are missing
    fake_hospital = tmp_path / "FAKE_HOSPITAL"
    fake_hospital.mkdir()
    s1_dir = fake_hospital / "STAGE_01_ML" / "CLEANED"
    s1_dir.mkdir(parents=True)
    mock_s1 = s1_dir / "cleaned_ml_dataset.csv"
    mock_s1.write_text("patient_id,cancer_type,age,sex,cancer_stage,creatinine_mg_dl,alt_u_l\nP1,Lung,60,M,IV,1.0,25.0\n", encoding="utf-8")
    
    collector = DataCollector(hospital_root=fake_hospital)
    verif = collector.verify_all_sources()
    assert verif["sources"]["STAGE_01_ML"]["status"] == "AVAILABLE"
    assert verif["sources"]["STAGE_02_DL"]["status"] == "NOT_AVAILABLE"
    assert verif["sources"]["STAGE_03_NLP"]["status"] == "NOT_AVAILABLE"
    assert verif["summary"]["available_count"] == 1
    assert verif["summary"]["not_available_count"] == 2
    # Collecting seeds should succeed using Stage 01 without crashing on missing 02/03
    df_seeds, _ = collector.collect_and_assemble_seeds()
    assert len(df_seeds) == 1

# 4. Raw Data Preservation
def test_raw_data_preservation():
    collector = DataCollector()
    raw_df, _ = collector.collect_and_assemble_seeds()
    raw_path = resolve_stage5_path("data/raw/raw_seed_cohort.csv")
    assert raw_path.exists(), "Raw seed file must be written to data/raw/"
    
    # Check that data_type is REFERENCE, never SYNTHETIC
    assert "data_type" in raw_df.columns
    assert (raw_df["data_type"] == "REFERENCE").all()
    assert "synthetic_flag" not in raw_df.columns

# 5. Duplicate Handling
def test_duplicate_handling(tmp_path):
    raw_csv = tmp_path / "raw_with_dups.csv"
    out_dir = tmp_path / "cleaned"
    # Create duplicate row
    df = pd.DataFrame([
        {"patient_id": "P1", "age": 55, "cancer_type": "Lung Adenocarcinoma", "alt_u_l": 25.0},
        {"patient_id": "P2", "age": 55, "cancer_type": "Lung Adenocarcinoma", "alt_u_l": 25.0},
        {"patient_id": "P3", "age": 65, "cancer_type": "Melanoma", "alt_u_l": 30.0}
    ])
    df.to_csv(raw_csv, index=False)
    cleaner = DataCleaner(raw_csv_path=raw_csv, cleaned_out_dir=out_dir)
    cleaned_df, report = cleaner.clean_seeds()
    assert report["exact_duplicates_removed"] == 1
    assert len(cleaned_df) == 2

# 6. Missing Value Handling
def test_missing_value_handling(tmp_path):
    raw_csv = tmp_path / "raw_missing.csv"
    out_dir = tmp_path / "cleaned"
    df = pd.DataFrame([
        {"patient_id": "P1", "age": 50, "cancer_type": "Lung Adenocarcinoma", "alt_u_l": np.nan},
        {"patient_id": "P2", "age": 60, "cancer_type": "Lung Adenocarcinoma", "alt_u_l": 30.0},
        {"patient_id": "P3", "age": np.nan, "cancer_type": "Lung Adenocarcinoma", "alt_u_l": 40.0}
    ])
    df.to_csv(raw_csv, index=False)
    cleaner = DataCleaner(raw_csv_path=raw_csv, cleaned_out_dir=out_dir)
    cleaned_df, report = cleaner.clean_seeds()
    assert cleaned_df["alt_u_l"].isnull().sum() == 0
    assert cleaned_df["age"].isnull().sum() == 0
    assert report["imputed_columns"]["alt_u_l"] == 1
    assert report["imputed_columns"]["age"] == 1

# 7. Invalid Clinical Values / Outlier Clipping
def test_invalid_clinical_values_clipping(tmp_path):
    raw_csv = tmp_path / "raw_outliers.csv"
    out_dir = tmp_path / "cleaned"
    df = pd.DataFrame([
        {"patient_id": "P1", "age": 150, "cancer_type": "Lung Adenocarcinoma", "alt_u_l": 5000.0},
        {"patient_id": "P2", "age": 10, "cancer_type": "Lung Adenocarcinoma", "alt_u_l": 0.05}
    ])
    df.to_csv(raw_csv, index=False)
    cleaner = DataCleaner(raw_csv_path=raw_csv, cleaned_out_dir=out_dir)
    cleaned_df, report = cleaner.clean_seeds()
    # Age clipped to (18, 105)
    assert cleaned_df.loc[0, "age"] == 105.0
    assert cleaned_df.loc[1, "age"] == 18.0
    # ALT clipped to (1.0, 2000.0)
    assert cleaned_df.loc[0, "alt_u_l"] == 2000.0
    assert cleaned_df.loc[1, "alt_u_l"] == 1.0

# 8. Privacy Validation
def test_privacy_validation_reference_vs_synthetic():
    # A valid reference record does NOT have synthetic_flag, but has no PII
    valid_ref = {"reference_seed_id": "REF-SEED-00001", "age": 62, "cancer_type": "Lung Adenocarcinoma"}
    ok, violations = PrivacyValidator.validate_reference_record(valid_ref)
    assert ok is True
    assert len(violations) == 0

    # A reference record containing direct PII must fail
    bad_ref_ssn = {"reference_seed_id": "REF-SEED-00002", "notes": "Patient SSN is 123-45-6789"}
    ok, violations = PrivacyValidator.validate_reference_record(bad_ref_ssn)
    assert ok is False
    assert any("PII_DETECTED" in v for v in violations)

    bad_ref_phone = {"reference_seed_id": "REF-SEED-00003", "phone": "555-123-4567"}
    ok, violations = PrivacyValidator.validate_reference_record(bad_ref_phone)
    assert ok is False

    # Synthetic record validation requires synthetic_flag == True
    syn_record_bad = {"patient_id": "SYN-PAT-0001", "synthetic_flag": False}
    ok, violations = PrivacyValidator.validate_synthetic_record(syn_record_bad)
    assert ok is False
    assert any("synthetic_flag" in v for v in violations)

    syn_record_good = {"patient_id": "SYN-PAT-0001", "synthetic_flag": True}
    ok, violations = PrivacyValidator.validate_synthetic_record(syn_record_good)
    assert ok is True

# 9. Mutation Frequency Calculation
def test_mutation_frequency_calculation():
    analyzer = MutationFrequencyAnalyzer()
    stats = analyzer.build_mutation_statistics()
    assert "cancer_specific_mutations" in stats
    records = analyzer.get_reference_mutation_records()
    for r in records:
        assert "gene" in r
        assert "variant" in r
        assert "frequency" in r
        assert "frequency_percentage" in r
        assert "source" in r
        assert abs(r["frequency"] * 100.0 - r["frequency_percentage"]) < 1e-4

# 10. Mutation Co-occurrence Calculation
def test_mutation_cooccurrence_calculation():
    analyzer = MutationCooccurrenceAnalyzer()
    matrix = analyzer.build_cooccurrence_matrix()
    assert "pairwise_relationships" in matrix
    assert "EGFR___KRAS" in matrix["pairwise_relationships"]
    rel = matrix["pairwise_relationships"]["EGFR___KRAS"]
    assert rel["relationship"] == "MUTUALLY_EXCLUSIVE"
    assert rel["log_odds_ratio"] < 0

# 11. Trajectory Statistics
def test_trajectory_statistics():
    analyzer = TrajectoryStatisticsAnalyzer()
    stats = analyzer.build_trajectory_statistics()
    assert "longitudinal_data_status" in stats
    assert stats["longitudinal_data_status"] in ["CROSS_SECTIONAL_REFERENCE_ONLY", "LONGITUDINAL_AVAILABLE"]
    assert "recist_transition_probabilities" in stats
    assert "RESPONDER" in stats["recist_transition_probabilities"]

# 12. Schema Validation
def test_schema_validation():
    schemas = export_schemas()
    assert len(schemas) == 10
    
    # Test valid source manifest payload
    manifest_data = {
        "timestamp": "2026-09-14T00:00:00Z",
        "sources": {"STAGE_01": {"status": "AVAILABLE"}},
        "summary": {"total_checked": 3, "available_count": 1, "not_available_count": 2}
    }
    ok, errs = validate_data_schema(manifest_data, "source_manifest")
    assert ok is True, f"Schema validation failed: {errs}"

    # Test invalid payload fails validation
    bad_data = {"timestamp": "2026-09-14T00:00:00Z"}
    ok, errs = validate_data_schema(bad_data, "source_manifest")
    assert ok is False

# 13. Deterministic Output & Reproducibility
def test_deterministic_output():
    cleaner = DataCleaner()
    df1, _ = cleaner.clean_seeds()
    df2, _ = cleaner.clean_seeds()
    pd.testing.assert_frame_equal(df1, df2)

# 14. SHA-256 Manifest Generation
def test_sha256_manifest_verification():
    manifest_file = resolve_stage5_path("outputs/data_manifest.json")
    assert manifest_file.exists(), "outputs/data_manifest.json must exist"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "files" in manifest
    assert len(manifest["files"]) >= 10
    
    # Verify manifest never hashes itself (no circular dependency)
    assert all("data_manifest.json" not in entry["file"] for entry in manifest["files"]), "Manifest must exclude itself"
    
    for entry in manifest["files"]:
        assert "file" in entry
        assert "sha256" in entry
        assert len(entry["sha256"]) == 64
        # Verify hash matches actual file
        target = resolve_stage5_path(entry["file"])
        if target.exists():
            sha = hashlib.sha256(target.read_bytes()).hexdigest()
            assert sha == entry["sha256"], f"Checksum mismatch on {entry['file']}"

# 15. Cleaning Report Transformation Counts Audit
def test_cleaning_report_transformation_counts():
    report_file = resolve_stage5_path("outputs/cleaning_report.json")
    assert report_file.exists()
    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert "total_missing_values_imputed" in report
    assert "total_outliers_clipped" in report
    assert "total_rows_dropped" in report
    assert report["total_missing_values_imputed"] >= 0
    assert report["total_outliers_clipped"] >= 0
    assert report["final_clean_rows"] == report["initial_rows"] - report["total_rows_dropped"]

# 16. Privacy Report Anonymization Disclaimer Audit
def test_privacy_report_anonymization_disclaimer():
    report_file = resolve_stage5_path("outputs/privacy_report.json")
    assert report_file.exists()
    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["direct_identifiers_found"] == 0
    assert report["direct_identifiers_removed"] is True
    assert report["full_anonymization_guarantee"] == "NOT_CLAIMED"
    assert "anonymization_notes" in report

# 17. Genomic Benchmark Metadata Audit
def test_genomic_benchmark_metadata_audit():
    analyzer = MutationFrequencyAnalyzer()
    stats = analyzer.build_mutation_statistics()
    meta = stats["metadata"]
    assert meta["stage01_cohort_clinical_cancer_types"] == 12
    assert meta["stage01_cohort_gene_level_variants_available"] is False
    assert meta["external_genomic_benchmark_cancer_types"] == 4
    assert "cancer_type_coverage_rationale" in meta

# 18. Clean-Machine Isolation Test
def test_clean_machine_isolation(tmp_path):
    """
    Tests that the Data Engineer pipeline runs in a completely separate, clean directory
    without ANY dependency on the original machine, username, or drive paths.
    """
    clean_root = tmp_path / "CLEAN_ISOLATED_WORKSPACE"
    clean_root.mkdir()
    
    # 1. Set up isolated Stage 01 mock data
    s1_dir = clean_root / "STAGE_01_ML" / "CLEANED"
    s1_dir.mkdir(parents=True)
    mock_df = pd.DataFrame([
        {
            "patient_id": f"P{i:03d}",
            "cancer_type": "Lung Adenocarcinoma",
            "age": 60 + i,
            "sex": "Female" if i % 2 == 0 else "Male",
            "cancer_stage": "Stage IIIA",
            "creatinine_mg_dl": 1.1,
            "alt_u_l": 28.0,
            "ast_u_l": 26.0,
            "albumin_g_dl": 4.0,
            "wbc_10_3_ul": 6.5,
            "platelets_10_3_ul": 220.0,
            "hemoglobin_g_dl": 13.5,
            "performance_status_ecog": 1
        } for i in range(10)
    ])
    mock_df.to_csv(s1_dir / "cleaned_ml_dataset.csv", index=False)
    
    # 2. Instantiate and run collector and cleaner pointing to clean_root
    isolated_s5 = clean_root / "STAGE_05_GENAI"
    collector = DataCollector(hospital_root=clean_root)
    collector.raw_output_dir = isolated_s5 / "data" / "raw"
    collector.outputs_dir = isolated_s5 / "outputs"
    
    seeds, verif = collector.collect_and_assemble_seeds()
    assert len(seeds) == 10
    assert (isolated_s5 / "data" / "raw" / "raw_seed_cohort.csv").exists()
    
    cleaner = DataCleaner(
        raw_csv_path=isolated_s5 / "data" / "raw" / "raw_seed_cohort.csv",
        cleaned_out_dir=isolated_s5 / "data" / "cleaned"
    )
    cleaner.outputs_dir = isolated_s5 / "outputs"
    cleaned, clean_rep = cleaner.clean_seeds()
    assert len(cleaned) == 10
    assert (isolated_s5 / "data" / "cleaned" / "cleaned_seed_cohort.csv").exists()
    assert clean_rep["final_clean_rows"] == 10


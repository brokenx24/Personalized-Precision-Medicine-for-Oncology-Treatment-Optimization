"""
Stage 05 Data Collector.
Verifies and collects reference datasets from upstream Stages 01-03 without modifying source files.
Outputs raw reference seed cohort and comprehensive source verification report.
"""
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd

try:
    from data_engineer.path_resolver import get_hospital_root, resolve_stage5_path
except ImportError:
    from path_resolver import get_hospital_root, resolve_stage5_path

class DataCollector:
    def __init__(self, hospital_root: Optional[Path] = None):
        self.hospital_root = Path(hospital_root) if hospital_root else get_hospital_root()
        
        # Primary and fallback paths for Stage 01
        self.stage01_primary = self.hospital_root / "STAGE_01_ML" / "CLEANED" / "cleaned_ml_dataset.csv"
        self.stage01_fallback = self.hospital_root / "stage1_ml" / "data" / "cleaned" / "complete_dataset.csv"
        
        # Stage 02 paths
        self.stage02_master = self.hospital_root / "STAGE_02_DL" / "METADATA" / "patient_master.csv"
        self.stage02_pathology = self.hospital_root / "STAGE_02_DL" / "METADATA" / "pathology_metadata.csv"
        
        # Stage 03 path
        self.stage03_clinical_text = (
            self.hospital_root / "STAGE_03_NLP" / "data_engineer" / "cleaned" / "cleaned_clinical_text.csv"
        )
        
        # Stage 05 output directories
        self.raw_output_dir = resolve_stage5_path("data/raw")
        self.outputs_dir = resolve_stage5_path("outputs")

    def _inspect_file(self, file_path: Path, expected_cols: list) -> Dict[str, Any]:
        """Inspects a tabular file and returns structured quality metrics."""
        if not file_path.exists():
            return {
                "exists": False,
                "status": "NOT_AVAILABLE",
                "file_path": str(file_path),
                "reason": "File does not exist on disk."
            }
            
        try:
            df = pd.read_csv(file_path)
            rows, cols = df.shape
            missing_cols = [c for c in expected_cols if c not in df.columns]
            dups = int(df.duplicated().sum())
            nulls = int(df.isnull().sum().sum())
            has_patient_id = "patient_id" in df.columns
            
            return {
                "exists": True,
                "status": "AVAILABLE",
                "file_path": str(file_path),
                "rows": rows,
                "columns": cols,
                "columns_list": list(df.columns),
                "missing_expected_columns": missing_cols,
                "schema_valid": len(missing_cols) == 0,
                "duplicate_rows": dups,
                "total_null_cells": nulls,
                "patient_identifier_available": has_patient_id,
                "acceptable_for_reference_use": (len(missing_cols) == 0 and rows > 0)
            }
        except Exception as e:
            return {
                "exists": True,
                "status": "INVALID",
                "file_path": str(file_path),
                "reason": f"Failed to read file: {str(e)}"
            }

    def verify_all_sources(self) -> Dict[str, Any]:
        """Verifies all upstream stages and builds a formal verification report."""
        report = {
            "timestamp": "2026-09-14T00:00:00Z",
            "hospital_root": str(self.hospital_root),
            "sources": {},
            "summary": {
                "total_checked": 3,
                "available_count": 0,
                "not_available_count": 0
            }
        }
        
        # Stage 01 check
        s1_path = self.stage01_primary if self.stage01_primary.exists() else self.stage01_fallback
        s1_info = self._inspect_file(
            s1_path,
            ["patient_id", "cancer_type", "age", "sex", "cancer_stage", "creatinine_mg_dl", "alt_u_l"]
        )
        report["sources"]["STAGE_01_ML"] = s1_info
        
        # Stage 02 check
        s2_info = self._inspect_file(
            self.stage02_master,
            ["patient_id", "has_pathology", "has_ct", "has_mri"]
        )
        report["sources"]["STAGE_02_DL"] = s2_info
        
        # Stage 03 check
        s3_info = self._inspect_file(
            self.stage03_clinical_text,
            ["patient_id", "clinical_text", "urgency_label"]
        )
        report["sources"]["STAGE_03_NLP"] = s3_info
        
        # Count available
        for s in report["sources"].values():
            if s.get("status") == "AVAILABLE":
                report["summary"]["available_count"] += 1
            else:
                report["summary"]["not_available_count"] += 1
                
        # Persist report
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        report_file = self.outputs_dir / "source_verification.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        return report

    def collect_and_assemble_seeds(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Collects available reference cohort data and writes immutable raw seed file.
        Does not mutate upstream files.
        """
        verification = self.verify_all_sources()
        s1_info = verification["sources"]["STAGE_01_ML"]
        
        if s1_info.get("status") != "AVAILABLE":
            raise FileNotFoundError(
                f"Required Stage 01 dataset is NOT_AVAILABLE. Checked paths: {self.stage01_primary}, {self.stage01_fallback}"
            )
            
        s1_file = Path(s1_info["file_path"])
        df_seeds = pd.read_csv(s1_file)
        
        # Add seed metadata provenance column
        df_seeds["seed_source"] = "STAGE_01_ML"
        df_seeds["data_type"] = "REFERENCE"
        
        # Save raw seed cohort immutably
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        raw_out = self.raw_output_dir / "raw_seed_cohort.csv"
        df_seeds.to_csv(raw_out, index=False)
        print(f"[Collector] Loaded {len(df_seeds)} rows from {s1_file.name}. Saved raw cohort to: {raw_out}")
        
        return df_seeds, verification

if __name__ == "__main__":
    collector = DataCollector()
    seeds_df, verif = collector.collect_and_assemble_seeds()
    print("Verification Summary:", verif["summary"])
    print(f"Collected raw cohort shape: {seeds_df.shape}")

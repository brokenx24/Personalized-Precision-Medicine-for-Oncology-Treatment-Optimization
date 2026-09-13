"""
Stage 05 Data Validation.
Executes schema validation, boundary checks, and privacy audits on cleaned reference seeds.
Produces formal validation report and privacy report.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd

try:
    from data_engineer.path_resolver import resolve_stage5_path
    from data_engineer.privacy_validator import PrivacyValidator
except ImportError:
    from path_resolver import resolve_stage5_path
    from privacy_validator import PrivacyValidator

class DataValidator:
    REQUIRED_COLUMNS = [
        "reference_seed_id", "data_type", "cancer_type", "age", "sex", "cancer_stage",
        "performance_status_ecog", "creatinine_mg_dl", "alt_u_l", "ast_u_l",
        "albumin_g_dl", "wbc_10_3_ul", "platelets_10_3_ul", "hemoglobin_g_dl"
    ]

    def __init__(self, cleaned_csv_path: Optional[Path] = None):
        self.cleaned_csv_path = Path(cleaned_csv_path) if cleaned_csv_path else resolve_stage5_path("data/cleaned/cleaned_seed_cohort.csv")
        self.outputs_dir = resolve_stage5_path("outputs")

    def validate(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        if not self.cleaned_csv_path.exists():
            raise FileNotFoundError(f"Cleaned dataset not found: {self.cleaned_csv_path}")

        df = pd.read_csv(self.cleaned_csv_path)
        
        # 1. Required columns check
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        
        # 2. Null checks across required columns
        null_counts = {col: int(df[col].isnull().sum()) for col in self.REQUIRED_COLUMNS if col in df.columns}
        total_nulls = sum(null_counts.values())

        # 3. Boundary validation
        from clean_data import DataCleaner
        boundary_violations = 0
        for col, (b_min, b_max) in DataCleaner.CLINICAL_BOUNDS.items():
            if col in df.columns:
                out_of_bounds = int(((df[col] < b_min) | (df[col] > b_max)).sum())
                boundary_violations += out_of_bounds

        # 4. Privacy Audit (REFERENCE data mode)
        privacy_ok, privacy_report = PrivacyValidator.audit_dataset(df, record_type="REFERENCE")

        # Compile validation report
        val_status = "PASS" if (len(missing_cols) == 0 and total_nulls == 0 and boundary_violations == 0 and privacy_ok) else "FAIL"
        
        validation_report = {
            "timestamp": "2026-09-14T00:00:00Z",
            "total_records_checked": len(df),
            "missing_required_columns": missing_cols,
            "null_value_violations": total_nulls,
            "boundary_violations": boundary_violations,
            "privacy_status": privacy_report["privacy_status"],
            "validation_status": val_status
        }

        # Save reports
        val_report_path = self.outputs_dir / "validation_report.json"
        with open(val_report_path, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=2)

        priv_report_path = self.outputs_dir / "privacy_report.json"
        with open(priv_report_path, "w", encoding="utf-8") as f:
            json.dump(privacy_report, f, indent=2)

        print(f"[Validator] Validation result: {val_status}. Reports written to {self.outputs_dir}")
        return validation_report, privacy_report

if __name__ == "__main__":
    validator = DataValidator()
    v_rep, p_rep = validator.validate()
    print("Validation status:", v_rep["validation_status"])
    print("Privacy status:", p_rep["privacy_status"])

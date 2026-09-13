"""
Stage 05 Data Cleaning & Normalization.
Performs deterministic cleaning, de-identification, imputation, and clinical outlier handling.
Produces cleaned reference cohort and formal transformation audit report.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

try:
    from data_engineer.path_resolver import resolve_stage5_path
except ImportError:
    from path_resolver import resolve_stage5_path

class DataCleaner:
    CLINICAL_BOUNDS = {
        "age": (18.0, 105.0),
        "weight_kg": (30.0, 200.0),
        "height_cm": (120.0, 220.0),
        "bmi": (12.0, 60.0),
        "creatinine_mg_dl": (0.1, 20.0),
        "alt_u_l": (1.0, 2000.0),
        "ast_u_l": (1.0, 2000.0),
        "albumin_g_dl": (0.5, 7.0),
        "wbc_10_3_ul": (0.1, 100.0),
        "platelets_10_3_ul": (5.0, 2000.0),
        "hemoglobin_g_dl": (3.0, 25.0),
        "bilirubin_mg_dl": (0.1, 30.0)
    }

    def __init__(self, raw_csv_path: Optional[Path] = None, cleaned_out_dir: Optional[Path] = None):
        self.raw_csv_path = Path(raw_csv_path) if raw_csv_path else resolve_stage5_path("data/raw/raw_seed_cohort.csv")
        self.cleaned_out_dir = Path(cleaned_out_dir) if cleaned_out_dir else resolve_stage5_path("data/cleaned")
        self.outputs_dir = resolve_stage5_path("outputs")

    def clean_seeds(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        self.cleaned_out_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.raw_csv_path.exists():
            raise FileNotFoundError(f"Raw seed cohort not found at: {self.raw_csv_path}")

        df = pd.read_csv(self.raw_csv_path)
        initial_rows = len(df)
        transformations = []

        # 1. Exact Duplicate Handling (excluding IDs)
        id_cols = [c for c in ["patient_id", "encounter_id", "seed_source", "data_type"] if c in df.columns]
        feature_cols = [c for c in df.columns if c not in id_cols]
        exact_dups = int(df.duplicated(subset=feature_cols).sum())
        if exact_dups > 0:
            df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
        transformations.append({
            "step": "exact_deduplication",
            "original_count": initial_rows,
            "removed_count": exact_dups,
            "modified_count": 0,
            "imputed_count": 0,
            "reason": "Remove identical cross-sectional patient encounters across all biological features"
        })

        # 2. De-identification / Reference ID assignment
        # We explicitly preserve reference distinction: data_type is REFERENCE, never synthetic.
        df["reference_seed_id"] = [f"REF-SEED-{i:05d}" for i in range(1, len(df) + 1)]
        df["data_type"] = "REFERENCE"
        transformations.append({
            "step": "de_identification",
            "original_count": len(df),
            "removed_count": 0,
            "modified_count": len(df),
            "imputed_count": 0,
            "reason": "Assign standardized non-PII reference identifier (REF-SEED-XXXXX) and label data_type as REFERENCE"
        })

        # 3. Missing Value Imputation using domain-specific median (numeric) and mode (categorical)
        imputed_counts = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            n_missing = int(df[col].isnull().sum())
            if n_missing > 0:
                imputed_counts[col] = n_missing
                med = float(df[col].median())
                df[col] = df[col].fillna(med)

        object_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in ["reference_seed_id", "patient_id"]]
        for col in object_cols:
            n_missing = int(df[col].isnull().sum())
            if n_missing > 0:
                imputed_counts[col] = n_missing
                mode_val = str(df[col].mode()[0]) if len(df[col].mode()) > 0 else "Unknown"
                df[col] = df[col].fillna(mode_val)

        transformations.append({
            "step": "missing_value_imputation",
            "original_count": len(df),
            "removed_count": 0,
            "modified_count": sum(imputed_counts.values()),
            "imputed_count": sum(imputed_counts.values()),
            "reason": "Impute missing numeric values via median and categorical via mode to ensure statistical completeness"
        })

        # 4. Outlier Clipping against Biological Plausibility Limits
        clipped_counts = {}
        for col, (b_min, b_max) in self.CLINICAL_BOUNDS.items():
            if col in df.columns:
                outliers = int(((df[col] < b_min) | (df[col] > b_max)).sum())
                if outliers > 0:
                    clipped_counts[col] = outliers
                    df[col] = df[col].clip(lower=b_min, upper=b_max)

        transformations.append({
            "step": "biological_outlier_clipping",
            "original_count": len(df),
            "removed_count": 0,
            "modified_count": sum(clipped_counts.values()),
            "imputed_count": 0,
            "reason": "Clip extreme values to physiological oncology boundaries to prevent unrealistic distortion"
        })

        # Build formal cleaning report
        total_imputed = sum(imputed_counts.values())
        total_clipped = sum(clipped_counts.values())
        cleaning_report = {
            "timestamp": "2026-09-14T00:00:00Z",
            "initial_rows": initial_rows,
            "final_clean_rows": len(df),
            "exact_duplicates_removed": exact_dups,
            "total_rows_dropped": exact_dups,
            "total_missing_values_imputed": total_imputed,
            "total_outliers_clipped": total_clipped,
            "imputed_columns": imputed_counts,
            "clipped_outliers": clipped_counts,
            "transformations": transformations
        }

        # Save cleaned dataset and report
        cleaned_path = self.cleaned_out_dir / "cleaned_seed_cohort.csv"
        df.to_csv(cleaned_path, index=False)
        
        report_path = self.outputs_dir / "cleaning_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(cleaning_report, f, indent=2)

        print(f"[Cleaner] Saved cleaned cohort ({len(df)} rows) to: {cleaned_path}")
        print(f"[Cleaner] Saved cleaning audit report to: {report_path}")
        return df, cleaning_report

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaned_df, report = cleaner.clean_seeds()
    print("Cleaning complete. Rows:", len(cleaned_df))

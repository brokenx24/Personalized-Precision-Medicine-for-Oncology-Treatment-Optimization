"""
Stage 05 Reference Distributions Builder.
Extracts empirical priors for demographics, stage, ECOG, and biomarkers across tumor types.
Outputs clinical and biomarker distributions grounded in actual reference data.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

try:
    from data_engineer.path_resolver import resolve_stage5_path
except ImportError:
    from path_resolver import resolve_stage5_path

class ReferenceDistributionBuilder:
    def __init__(self, cleaned_csv: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.cleaned_csv = Path(cleaned_csv) if cleaned_csv else resolve_stage5_path("data/cleaned/cleaned_seed_cohort.csv")
        self.output_dir = Path(output_dir) if output_dir else resolve_stage5_path("data/reference")
        self.legacy_output_dir = resolve_stage5_path("data/reference_distributions")

    def _compute_numeric_stats(self, series: pd.Series) -> Dict[str, float]:
        """Computes complete statistical profile for a numeric column."""
        s = series.dropna()
        if len(s) == 0:
            return {}
        p25 = float(s.quantile(0.25))
        p75 = float(s.quantile(0.75))
        return {
            "count": int(len(s)),
            "missingness_rate": float(series.isnull().mean()),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "q1": p25,
            "q3": p75,
            "iqr": float(p75 - p25),
            "p10": float(s.quantile(0.10)),
            "p50": float(s.median()),
            "p90": float(s.quantile(0.90)),
            "p99": float(s.quantile(0.99))
        }

    def build_distributions(self) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.cleaned_csv.exists():
            raise FileNotFoundError(f"Cleaned cohort file not found: {self.cleaned_csv}")

        df = pd.read_csv(self.cleaned_csv)
        now_iso = "2026-09-14T00:00:00Z"

        # 1. Clinical Distributions
        clinical_dists = {
            "metadata": {
                "generated_at": now_iso,
                "version": "1.0.0",
                "cohort_size": len(df),
                "source": "Stage 01 Reference Cleaned Seed Cohort"
            },
            "cancer_type_proportions": df["cancer_type"].value_counts(normalize=True).to_dict(),
            "demographics": {
                "age": self._compute_numeric_stats(df["age"]),
                "sex_ratio": df["sex"].value_counts(normalize=True).to_dict(),
                "ecog_distribution": df["performance_status_ecog"].value_counts(normalize=True).to_dict() if "performance_status_ecog" in df.columns else {},
                "stage_distribution": df["cancer_stage"].value_counts(normalize=True).to_dict() if "cancer_stage" in df.columns else {},
                "risk_class_distribution": df["oncology_risk_class"].value_counts(normalize=True).to_dict() if "oncology_risk_class" in df.columns else {}
            },
            "cancer_specific_profiles": {}
        }

        for cancer_type, group in df.groupby("cancer_type"):
            clinical_dists["cancer_specific_profiles"][cancer_type] = {
                "count": len(group),
                "age": self._compute_numeric_stats(group["age"]),
                "stage_proportions": group["cancer_stage"].value_counts(normalize=True).to_dict() if "cancer_stage" in group.columns else {},
                "ecog_proportions": group["performance_status_ecog"].value_counts(normalize=True).to_dict() if "performance_status_ecog" in group.columns else {}
            }

        # 2. Biomarker / Laboratory Distributions
        # Only analyze numeric columns that ACTUALLY exist in the reference cohort
        available_numeric = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ["age", "weight_kg", "height_cm", "bmi"]
        ]
        
        biomarker_dists = {
            "metadata": {
                "generated_at": now_iso,
                "version": "1.0.0",
                "cohort_size": len(df),
                "numeric_features_analyzed": len(available_numeric)
            },
            "laboratory_distributions": {}
        }

        for col in available_numeric:
            biomarker_dists["laboratory_distributions"][col] = self._compute_numeric_stats(df[col])

        # Write clinical distributions
        clin_file = self.output_dir / "clinical_distributions.json"
        with open(clin_file, "w", encoding="utf-8") as f:
            json.dump(clinical_dists, f, indent=2)

        # Write biomarker distributions
        bio_file = self.output_dir / "biomarker_distributions.json"
        with open(bio_file, "w", encoding="utf-8") as f:
            json.dump(biomarker_dists, f, indent=2)

        # Write unified reference_distributions.json for backwards compatibility
        unified = {
            "metadata": clinical_dists["metadata"],
            "cancer_type_proportions": clinical_dists["cancer_type_proportions"],
            "overall_demographics": clinical_dists["demographics"],
            "cancer_specific_distributions": clinical_dists["cancer_specific_profiles"],
            "laboratory_distributions": biomarker_dists["laboratory_distributions"]
        }
        legacy_file = self.legacy_output_dir / "reference_distributions.json"
        with open(legacy_file, "w", encoding="utf-8") as f:
            json.dump(unified, f, indent=2)

        print(f"[Distributions] Clinical distributions saved to: {clin_file}")
        print(f"[Distributions] Biomarker distributions saved to: {bio_file}")
        print(f"[Distributions] Unified distributions saved to: {legacy_file}")

        return {
            "clinical": clinical_dists,
            "biomarkers": biomarker_dists,
            "unified": unified
        }

if __name__ == "__main__":
    builder = ReferenceDistributionBuilder()
    res = builder.build_distributions()
    print("Clinical distribution keys:", list(res["clinical"].keys()))
    print("Biomarker labs analyzed:", len(res["biomarkers"]["laboratory_distributions"]))

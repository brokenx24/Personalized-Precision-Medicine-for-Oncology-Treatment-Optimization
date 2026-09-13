"""
Synthetic Patient Profile Generator.
Samples demographic and clinical baseline features from empirical reference distributions.
"""
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

HOSPITAL_ROOT = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL")

class PatientGenerator:
    def __init__(self, distributions_path: Optional[Path] = None, seed: int = 42):
        self.distributions_path = distributions_path or (
            HOSPITAL_ROOT / "STAGE_05_GENAI" / "data" / "reference_distributions" / "reference_distributions.json"
        )
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.distributions = self._load_distributions()

    def _load_distributions(self) -> Dict[str, Any]:
        if not self.distributions_path.exists():
            raise FileNotFoundError(f"Reference distributions missing at {self.distributions_path}")
        with open(self.distributions_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_patient(self, patient_idx: int = 1, cancer_type: Optional[str] = None, stage: Optional[str] = None) -> Dict[str, Any]:
        patient_id = f"SYN-PAT-{patient_idx:05d}"
        
        # 1. Cancer Type sampling
        if not cancer_type:
            types = list(self.distributions["cancer_type_proportions"].keys())
            weights = list(self.distributions["cancer_type_proportions"].values())
            cancer_type = random.choices(types, weights=weights, k=1)[0]

        # 2. Stage sampling
        spec = self.distributions["cancer_specific_distributions"].get(cancer_type, {})
        if not stage:
            stages = list(spec.get("stage_proportions", {"Stage IV": 0.5, "Stage III": 0.3, "Stage II": 0.15, "Stage I": 0.05}).keys())
            s_weights = list(spec.get("stage_proportions", {"Stage IV": 0.5, "Stage III": 0.3, "Stage II": 0.15, "Stage I": 0.05}).values())
            stage = random.choices(stages, weights=s_weights, k=1)[0]

        # 3. Demographics
        age_mean = spec.get("age_mean", 64.0)
        age_std = spec.get("age_std", 9.5)
        age = int(np.clip(np.random.normal(age_mean, age_std), 25, 88))
        
        sex_ratio = self.distributions["overall_demographics"].get("sex_ratio", {"Male": 0.52, "Female": 0.48})
        valid_sexes = [s for s in sex_ratio.keys() if s in ["Male", "Female"]]
        valid_weights = [sex_ratio[s] for s in valid_sexes]
        if not valid_sexes:
            valid_sexes, valid_weights = ["Male", "Female"], [0.5, 0.5]
        sex = random.choices(valid_sexes, weights=valid_weights, k=1)[0]

        ecog_dist = spec.get("ecog_proportions", {0: 0.35, 1: 0.45, 2: 0.15, 3: 0.05})
        ecog = int(random.choices(list(ecog_dist.keys()), weights=list(ecog_dist.values()), k=1)[0])

        height = round(float(np.clip(np.random.normal(170.0 if sex == "Male" else 160.0, 8.0), 135.0, 205.0)), 1)
        weight = round(float(np.clip(np.random.normal(78.0 if sex == "Male" else 65.0, 14.0), 38.0, 155.0)), 1)
        bmi = round(float(np.clip(weight / ((height / 100.0) ** 2), 14.5, 48.0)), 1)

        # 4. Baseline Laboratories
        labs = {
            "creatinine_mg_dl": round(float(np.clip(np.random.normal(1.0, 0.25), 0.5, 3.5)), 2),
            "alt_u_l": round(float(np.clip(np.random.normal(28.0, 12.0), 10.0, 180.0)), 1),
            "ast_u_l": round(float(np.clip(np.random.normal(26.0, 11.0), 10.0, 160.0)), 1),
            "albumin_g_dl": round(float(np.clip(np.random.normal(4.1, 0.4), 2.1, 5.0)), 2),
            "wbc_10_3_ul": round(float(np.clip(np.random.normal(6.8, 1.8), 2.0, 22.0)), 2),
            "platelets_10_3_ul": round(float(np.clip(np.random.normal(240.0, 55.0), 50.0, 600.0)), 1),
            "hemoglobin_g_dl": round(float(np.clip(np.random.normal(13.2, 1.6), 7.5, 17.5)), 1)
        }

        # 5. Baseline Biomarkers
        cea = round(float(np.clip(np.random.exponential(4.0), 0.5, 60.0)), 2)
        ca125 = round(float(np.clip(np.random.exponential(25.0), 5.0, 350.0)), 1)
        psa = round(float(np.clip(np.random.exponential(6.0), 0.1, 80.0)), 2) if sex == "Male" else 0.0
        ctdna = round(float(np.clip(np.random.exponential(2.5), 0.1, 25.0)), 2)

        biomarkers = {
            "cea_ng_ml": cea,
            "ca125_u_ml": ca125,
            "psa_ng_ml": psa,
            "ctdna_maf_pct": ctdna
        }

        return {
            "patient_id": patient_id,
            "synthetic_flag": True,
            "is_hypothetical": False,
            "age": age,
            "sex": sex,
            "cancer_type": cancer_type,
            "cancer_stage": stage,
            "tumor_grade": "G2",
            "ecog_performance_status": ecog,
            "weight_kg": weight,
            "height_cm": height,
            "bmi": bmi,
            "baseline_biomarkers": biomarkers,
            "baseline_labs": labs
        }

if __name__ == "__main__":
    pg = PatientGenerator()
    p = pg.generate_patient(1)
    print("Sample synthetic patient:", json.dumps(p, indent=2))

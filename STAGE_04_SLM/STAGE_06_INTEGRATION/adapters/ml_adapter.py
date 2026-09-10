"""
STAGE 06 INTEGRATION - ML ADAPTER
Adapts patient dictionary to Stage 01 Feature Engineer and XGBoost classifier.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

try:
    from model_registry.ml_registry import MLModelRegistry
except ImportError:
    from STAGE_06_INTEGRATION.model_registry.ml_registry import MLModelRegistry

class MLAdapter:
    DEFAULT_CATEGORICALS = {
        "cancer_type": "Lung Adenocarcinoma",
        "sex": "Male",
        "cancer_stage": "Stage IV",
        "tumor_grade": "G2",
        "path_t_stage": "T2",
        "path_n_stage": "N1",
        "path_m_stage": "M0"
    }

    def __init__(self, ml_registry: MLModelRegistry = None):
        self.ml_registry = ml_registry or MLModelRegistry()
        
    def load_model(self):
        self.ml_registry.load()
        
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        self.load_model()
        model = self.ml_registry.model
        scaler = self.ml_registry.scaler
        encoder = self.ml_registry.encoder
        num_cols = self.ml_registry.num_cols
        cat_cols = self.ml_registry.cat_cols
        medians = self.ml_registry.train_medians
        
        # Build normalized numeric row
        row = {}
        for col in num_cols:
            row[col] = float(patient_data.get(col, medians.get(col, 0.0)))
            
        for col in cat_cols:
            row[col] = str(patient_data.get(col, self.DEFAULT_CATEGORICALS.get(col, "UNKNOWN")))
            
        # Map known synonyms
        if "gender" in patient_data and "sex" not in patient_data:
            g = patient_data["gender"]
            row["sex"] = "Male" if g in [1, "1", "M", "Male"] else "Female"
        if "ecog_performance_status" in patient_data:
            row["performance_status_ecog"] = float(patient_data["ecog_performance_status"])
        if "serum_creatinine_mg_dl" in patient_data:
            row["creatinine_mg_dl"] = float(patient_data["serum_creatinine_mg_dl"])
        if "white_blood_cell_k_ul" in patient_data:
            row["wbc_10_3_ul"] = float(patient_data["white_blood_cell_k_ul"])
        if "platelet_count_k_ul" in patient_data:
            row["platelets_10_3_ul"] = float(patient_data["platelet_count_k_ul"])
        if "tmb_mutations_mb" in patient_data:
            row["tmb_nonsynonymous"] = float(patient_data["tmb_mutations_mb"])

        df = pd.DataFrame([row])
        
        # Calculate derived features
        df["ast_alt_ratio"] = df["ast_u_l"] / (df["alt_u_l"] + 1e-5)
        df["alb_creat_ratio"] = df["albumin_g_dl"] / (df["creatinine_mg_dl"] + 1e-5)
        df["systemic_immune_index"] = (df["platelets_10_3_ul"] * df["wbc_10_3_ul"]) / (df["hemoglobin_g_dl"] + 1e-5)
        df["biomarker_hypoxia_burden"] = df["ctdna_baseline_maf"] * (df["buffa_hypoxia_score"].fillna(0) + 1.0)
        df["genomic_instability_index"] = df["fraction_genome_altered"].fillna(0) * (df["aneuploidy_score"].fillna(0) + 1.0)
        
        # Transform through trained scaler and encoder
        scaled_num = scaler.transform(df[num_cols])
        encoded_cat = encoder.transform(df[cat_cols])
        
        X = np.hstack([scaled_num, encoded_cat]).astype(np.float32)
        
        # Predict probabilities
        probs = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = self.ml_registry.classes[pred_idx]
        confidence = float(probs[pred_idx])
        risk_score = float(probs[1] * 0.5 + probs[2] * 1.0)
        
        return {
            "risk_class": pred_class,
            "risk_score": round(risk_score, 4),
            "confidence": round(confidence, 4),
            "class_probabilities": {
                "LOW": round(float(probs[0]), 4),
                "MODERATE": round(float(probs[1]), 4),
                "HIGH": round(float(probs[2]), 4)
            }
        }

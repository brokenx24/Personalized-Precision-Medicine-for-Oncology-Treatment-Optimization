"""
STAGE 06 INTEGRATION - ML REGISTRY
Loads and caches Stage 01 Best ML Model (XGBoost) and exact feature preprocessors.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Provide ClinicalFeatureEngineer definition shim for joblib unpickling
class ClinicalFeatureEngineer:
    pass

setattr(sys.modules["__main__"], "ClinicalFeatureEngineer", ClinicalFeatureEngineer)

class MLModelRegistry:
    def __init__(self, hospital_root: Path = None):
        if hospital_root is None:
            hospital_root = Path(__file__).resolve().parent.parent.parent
        self.hospital_root = hospital_root
        self.model_path = self.hospital_root / "STAGE_01_ML" / "MODELS" / "best_ml_model.joblib"
        self.fe_path = self.hospital_root / "STAGE_01_ML" / "FEATURES" / "clinical_feature_engineer.joblib"
        self.complete_csv = self.hospital_root / "STAGE_01_ML" / "CLEANED" / "complete_imputed_dataset.csv"
        self.splits_train_csv = self.hospital_root / "STAGE_01_ML" / "SPLITS" / "train.csv"
        
        self.model = None
        self.feature_engineer = None
        self.scaler = None
        self.encoder = None
        self.num_cols = None
        self.cat_cols = None
        self.train_medians = {}
        self.is_loaded = False
        self.version = "1.0.0-STAGE_01_ML"
        self.classes = ["LOW", "MODERATE", "HIGH"]
        
    def load(self):
        if not self.is_loaded:
            setattr(sys.modules["__main__"], "ClinicalFeatureEngineer", ClinicalFeatureEngineer)
            if not self.model_path.exists():
                raise FileNotFoundError(f"ML model missing at {self.model_path}")
            self.model = joblib.load(str(self.model_path))
            
            if self.fe_path.exists():
                try:
                    self.feature_engineer = joblib.load(str(self.fe_path))
                except Exception:
                    self.feature_engineer = None
                    
            # Fit exact 96-feature scaler and encoder from training split
            df = pd.read_csv(str(self.complete_csv))
            tr_pts = set(pd.read_csv(str(self.splits_train_csv))["patient_id"])
            df_train = df[df["patient_id"].isin(tr_pts)].copy()
            
            # Interaction features
            df_train["ast_alt_ratio"] = df_train["ast_u_l"] / (df_train["alt_u_l"] + 1e-5)
            df_train["alb_creat_ratio"] = df_train["albumin_g_dl"] / (df_train["creatinine_mg_dl"] + 1e-5)
            df_train["systemic_immune_index"] = (df_train["platelets_10_3_ul"] * df_train["wbc_10_3_ul"]) / (df_train["hemoglobin_g_dl"] + 1e-5)
            df_train["biomarker_hypoxia_burden"] = df_train["ctdna_baseline_maf"] * (df_train["buffa_hypoxia_score"].fillna(0) + 1.0)
            df_train["genomic_instability_index"] = df_train["fraction_genome_altered"].fillna(0) * (df_train["aneuploidy_score"].fillna(0) + 1.0)
            
            drop_cols = ["patient_id", "encounter_id", "study_id", "standardized_date", "oncology_risk_class"]
            feature_cols = [c for c in df_train.columns if c not in drop_cols]
            
            self.num_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            self.cat_cols = df_train[feature_cols].select_dtypes(include=["object"]).columns.tolist()
            
            self.train_medians = df_train[self.num_cols].median().to_dict()
            
            self.scaler = StandardScaler().fit(df_train[self.num_cols])
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df_train[self.cat_cols])
            
            self.is_loaded = True
        return self

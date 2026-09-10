import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class ClinicalFeatureEngineer:
    def __init__(self):
        self.num_imputer = None
        self.scaler = None
        self.fitted = False
        self.num_cols = None
        self.cat_cols = None
        self.feature_names = None
        
    def fit(self, df_train):
        df = df_train.copy()
        
        # 1. Engineer interaction / derived clinical features
        # AST/ALT ratio (De Ritis ratio: clinical indicator of liver injury / metabolic stress)
        df['ast_alt_ratio'] = df['ast_u_l'] / (df['alt_u_l'] + 1e-5)
        # Albumin-to-Creatinine surrogate ratio
        df['alb_creat_ratio'] = df['albumin_g_dl'] / (df['creatinine_mg_dl'] + 1e-5)
        # Systemic Immune-Inflammation Index surrogate (Platelets * WBC / Hemoglobin)
        df['systemic_immune_index'] = (df['platelets_10_3_ul'] * df['wbc_10_3_ul']) / (df['hemoglobin_g_dl'] + 1e-5)
        # Biomarker to Hypoxia interaction
        df['biomarker_hypoxia_burden'] = df['ctdna_baseline_maf'] * (df['buffa_hypoxia_score'].fillna(0) + 1.0)
        # Genomic Instability Index (Fraction Genome Altered * Aneuploidy Score)
        df['genomic_instability_index'] = df['fraction_genome_altered'].fillna(0) * (df['aneuploidy_score'].fillna(0) + 1.0)
        
        # Identify numeric & categorical candidate features
        target_col = 'oncology_risk_class'
        drop_cols = ['patient_id', 'encounter_id', 'study_id', 'standardized_date', target_col]
        
        feature_cols = [c for c in df.columns if c not in drop_cols]
        self.num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
        
        # Fit Median Imputer on Training Numeric Features
        self.num_imputer = SimpleImputer(strategy='median')
        self.num_imputer.fit(df[self.num_cols])
        
        # Fit Most Frequent Imputer on Training Categorical Features
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.cat_imputer.fit(df[self.cat_cols])
        
        # Fit Scaler on Training Numeric Features
        imputed_num = self.num_imputer.transform(df[self.num_cols])
        self.scaler = StandardScaler()
        self.scaler.fit(imputed_num)
        
        # Fit OneHotEncoder on Training Categorical Features
        imputed_cat = self.cat_imputer.transform(df[self.cat_cols])
        self.cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.cat_encoder.fit(imputed_cat)
        
        cat_feature_names = self.cat_encoder.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names = self.num_cols + cat_feature_names
        self.fitted = True
        
        print(f"Feature Engineer Fitted on Training Set:")
        print(f"  Numeric features ({len(self.num_cols)}): {self.num_cols[:8]}...")
        print(f"  Categorical features ({len(self.cat_cols)}): {self.cat_cols}")
        print(f"  Total transformed feature dimensions: {len(self.feature_names)}")
        return self
        
    def transform(self, df_input):
        if not self.fitted:
            raise ValueError("Feature Engineer must be fitted before transform!")
            
        df = df_input.copy()
        # Compute derived features using exact same logic
        df['ast_alt_ratio'] = df['ast_u_l'] / (df['alt_u_l'] + 1e-5)
        df['alb_creat_ratio'] = df['albumin_g_dl'] / (df['creatinine_mg_dl'] + 1e-5)
        df['systemic_immune_index'] = (df['platelets_10_3_ul'] * df['wbc_10_3_ul']) / (df['hemoglobin_g_dl'] + 1e-5)
        df['biomarker_hypoxia_burden'] = df['ctdna_baseline_maf'] * (df['buffa_hypoxia_score'].fillna(0) + 1.0)
        df['genomic_instability_index'] = df['fraction_genome_altered'].fillna(0) * (df['aneuploidy_score'].fillna(0) + 1.0)
        
        # Impute & scale numeric
        imputed_num = self.num_imputer.transform(df[self.num_cols])
        scaled_num = self.scaler.transform(imputed_num)
        
        # Impute & encode categorical
        imputed_cat = self.cat_imputer.transform(df[self.cat_cols])
        encoded_cat = self.cat_encoder.transform(imputed_cat)
        
        # Combine
        X_trans = np.hstack([scaled_num, encoded_cat])
        return X_trans

def engineer_stage1_features(splits_dir, features_dir):
    print("=" * 70, flush=True)
    print("STAGE 1: FEATURE ENGINEERING PIPELINE (FITTED ON TRAIN ONLY)", flush=True)
    print("=" * 70, flush=True)
    
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "validation.csv")
    test_path = os.path.join(splits_dir, "test.csv")
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    fe = ClinicalFeatureEngineer()
    fe.fit(df_train)
    
    # Target encoding: LOW -> 0, MODERATE -> 1, HIGH -> 2
    label_mapping = {'LOW': 0, 'MODERATE': 1, 'HIGH': 2}
    
    X_train = fe.transform(df_train)
    y_train = df_train['oncology_risk_class'].map(label_mapping).values
    
    X_val = fe.transform(df_val)
    y_val = df_val['oncology_risk_class'].map(label_mapping).values
    
    X_test = fe.transform(df_test)
    y_test = df_test['oncology_risk_class'].map(label_mapping).values
    
    os.makedirs(features_dir, exist_ok=True)
    joblib.dump(fe, os.path.join(features_dir, "clinical_feature_engineer.joblib"))
    
    np.savez_compressed(
        os.path.join(features_dir, "stage1_processed_arrays.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=np.array(fe.feature_names)
    )
    
    print(f"Processed arrays saved to: {os.path.join(features_dir, 'stage1_processed_arrays.npz')}")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    print(f"  X_test : {X_test.shape}, y_test : {y_test.shape}")
    print("=" * 70, flush=True)
    return fe, X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    p_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s_dir = os.path.join(p_root, "SPLITS")
    f_dir = os.path.join(p_root, "FEATURES")
    engineer_stage1_features(s_dir, f_dir)

"""
================================================================================
CLINICAL FEATURE ENGINEERING & LEAK-FREE TRANSFORMATION PIPELINE
================================================================================
Creates non-linear clinical interaction features, performs strict patient-level
data partitioning, and fits feature scalers and encoders strictly on the training
split to prevent any data leakage.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

def compute_interaction_features(df):
    d = df.copy()
    d['ast_alt_ratio'] = d['ast_u_l'] / (d['alt_u_l'] + 1e-5)
    d['alb_creat_ratio'] = d['albumin_g_dl'] / (d['creatinine_mg_dl'] + 1e-5)
    d['systemic_immune_index'] = (d['platelets_10_3_ul'] * d['wbc_10_3_ul']) / (d['hemoglobin_g_dl'] + 1e-5)
    d['biomarker_hypoxia_burden'] = d['ctdna_baseline_maf'] * (d['buffa_hypoxia_score'] + 1.0)
    d['genomic_instability_index'] = d['fraction_genome_altered'] * (d['aneuploidy_score'] + 1.0)
    return d

def run_feature_engineering(project_root=None):
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
    data_path = os.path.join(project_root, "stage1_ml", "data", "cleaned", "complete_dataset.csv")
    features_dir = os.path.join(project_root, "stage1_ml", "features")
    models_dir = os.path.join(project_root, "stage1_ml", "models")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 70)
    print("STAGE 1 FEATURE ENGINEERING & PATIENT-LEVEL PARTITIONING")
    print("=" * 70)
    df = pd.read_csv(data_path)
    
    # 1. Patient-Level Splitting (70% Train, 15% Val, 15% Test)
    existing_train_split = os.path.join(project_root, "STAGE_01_ML", "SPLITS", "train.csv")
    if os.path.exists(existing_train_split):
        train_pts = set(pd.read_csv(existing_train_split)['patient_id'])
        val_pts = set(pd.read_csv(os.path.join(project_root, "STAGE_01_ML", "SPLITS", "validation.csv"))['patient_id'])
        test_pts = set(pd.read_csv(os.path.join(project_root, "STAGE_01_ML", "SPLITS", "test.csv"))['patient_id'])
    else:
        unique_pts = df['patient_id'].unique()
        np.random.seed(42)
        shuffled = np.random.permutation(unique_pts)
        n_tr = int(len(shuffled) * 0.70)
        n_va = int(len(shuffled) * 0.15)
        train_pts = set(shuffled[:n_tr])
        val_pts = set(shuffled[n_tr:n_tr+n_va])
        test_pts = set(shuffled[n_tr+n_va:])
        
    df_train = df[df['patient_id'].isin(train_pts)].copy()
    df_val = df[df['patient_id'].isin(val_pts)].copy()
    df_test = df[df['patient_id'].isin(test_pts)].copy()
    
    # Leakage Assertion
    assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage detected between Train and Val!"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage detected between Train and Test!"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage detected between Val and Test!"
    
    print(f"Zero-Leakage Patient Splits:")
    print(f"  Train Split : {len(df_train)} patients (70%)")
    print(f"  Val Split   : {len(df_val)} patients (15%)")
    print(f"  Test Split  : {len(df_test)} patients (15%)")
    
    # Compute clinical interactions
    df_train = compute_interaction_features(df_train)
    df_val = compute_interaction_features(df_val)
    df_test = compute_interaction_features(df_test)
    
    target_col = 'oncology_risk_class'
    drop_cols = ['patient_id', 'encounter_id', 'study_id', 'standardized_date', target_col]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]
    
    num_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_train[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    # Fit strictly on Training Split
    scaler = StandardScaler()
    scaler.fit(df_train[num_cols])
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(df_train[cat_cols])
    
    encoded_cat_names = encoder.get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + encoded_cat_names
    
    def transform_split(d):
        s_num = scaler.transform(d[num_cols])
        e_cat = encoder.transform(d[cat_cols])
        return np.hstack([s_num, e_cat])
        
    X_train = np.ascontiguousarray(transform_split(df_train), dtype=np.float32)
    X_val = np.ascontiguousarray(transform_split(df_val), dtype=np.float32)
    X_test = np.ascontiguousarray(transform_split(df_test), dtype=np.float32)
    
    label_map = {'LOW': 0, 'MODERATE': 1, 'HIGH': 2}
    y_train = np.ascontiguousarray(df_train['oncology_risk_class'].map(label_map).values, dtype=np.int32)
    y_val = np.ascontiguousarray(df_val['oncology_risk_class'].map(label_map).values, dtype=np.int32)
    y_test = np.ascontiguousarray(df_test['oncology_risk_class'].map(label_map).values, dtype=np.int32)
    
    # 3. Save Serialized Preprocessor Dictionary (Pure scikit-learn objects)
    preproc_dict = {
        'scaler': scaler,
        'encoder': encoder,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'feature_names': feature_names
    }
    preproc_path = os.path.join(models_dir, "feature_preprocessor.joblib")
    joblib.dump(preproc_dict, preproc_path)
    print(f"\nSaved Preprocessor Artifact to: {preproc_path}")
    
    # 4. Save Processed Feature Arrays
    arrays_path = os.path.join(features_dir, "processed_features.npz")
    np.savez_compressed(
        arrays_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=np.array(feature_names)
    )
    print(f"Saved Processed Arrays to: {arrays_path}")
    print(f"  Transformed Dimension: {X_train.shape[1]} features")
    print("=" * 70)
    return preproc_dict, X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    run_feature_engineering()

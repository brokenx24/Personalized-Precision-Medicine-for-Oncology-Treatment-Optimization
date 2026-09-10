import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def prepare_completed_features(project_root="."):
    stage1_dir = os.path.join(project_root, "STAGE_01_ML")
    complete_csv = os.path.join(stage1_dir, "CLEANED", "complete_imputed_dataset.csv")
    splits_dir = os.path.join(stage1_dir, "SPLITS")
    features_dir = os.path.join(stage1_dir, "FEATURES")
    
    df = pd.read_csv(complete_csv)
    train_pts = set(pd.read_csv(os.path.join(splits_dir, "train.csv"))['patient_id'])
    val_pts = set(pd.read_csv(os.path.join(splits_dir, "validation.csv"))['patient_id'])
    test_pts = set(pd.read_csv(os.path.join(splits_dir, "test.csv"))['patient_id'])
    
    df_train = df[df['patient_id'].isin(train_pts)].copy()
    df_val = df[df['patient_id'].isin(val_pts)].copy()
    df_test = df[df['patient_id'].isin(test_pts)].copy()
    
    # Interaction features
    for d in [df_train, df_val, df_test]:
        d['ast_alt_ratio'] = d['ast_u_l'] / (d['alt_u_l'] + 1e-5)
        d['alb_creat_ratio'] = d['albumin_g_dl'] / (d['creatinine_mg_dl'] + 1e-5)
        d['systemic_immune_index'] = (d['platelets_10_3_ul'] * d['wbc_10_3_ul']) / (d['hemoglobin_g_dl'] + 1e-5)
        d['biomarker_hypoxia_burden'] = d['ctdna_baseline_maf'] * (d['buffa_hypoxia_score'] + 1.0)
        d['genomic_instability_index'] = d['fraction_genome_altered'] * (d['aneuploidy_score'] + 1.0)
        
    target_col = 'oncology_risk_class'
    drop_cols = ['patient_id', 'encounter_id', 'study_id', 'standardized_date', target_col]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]
    
    num_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_train[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    scaler = StandardScaler()
    scaler.fit(df_train[num_cols])
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(df_train[cat_cols])
    
    def transform_split(data_split):
        scaled = scaler.transform(data_split[num_cols])
        encoded = encoder.transform(data_split[cat_cols])
        return np.hstack([scaled, encoded])
        
    X_train = np.ascontiguousarray(transform_split(df_train), dtype=np.float32)
    X_val = np.ascontiguousarray(transform_split(df_val), dtype=np.float32)
    X_test = np.ascontiguousarray(transform_split(df_test), dtype=np.float32)
    
    label_map = {'LOW': 0, 'MODERATE': 1, 'HIGH': 2}
    y_train = np.ascontiguousarray(df_train[target_col].map(label_map).values, dtype=np.int32)
    y_val = np.ascontiguousarray(df_val[target_col].map(label_map).values, dtype=np.int32)
    y_test = np.ascontiguousarray(df_test[target_col].map(label_map).values, dtype=np.int32)
    
    feature_names = num_cols + encoder.get_feature_names_out(cat_cols).tolist()
    
    out_npz = os.path.join(features_dir, "completed_arrays.npz")
    np.savez_compressed(
        out_npz,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=np.array(feature_names)
    )
    print(f"Saved completed dataset arrays: {out_npz}")
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

if __name__ == "__main__":
    p_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prepare_completed_features(p_root)

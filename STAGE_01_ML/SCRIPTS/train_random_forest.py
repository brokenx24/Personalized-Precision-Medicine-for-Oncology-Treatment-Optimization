import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, recall_score

def train_rf(project_root="."):
    print("[1/3] Tuning Regularized Random Forest in isolated runner...", flush=True)
    features_npz = os.path.join(project_root, "STAGE_01_ML", "FEATURES", "stage1_processed_arrays.npz")
    models_dir = os.path.join(project_root, "STAGE_01_ML", "MODELS")
    os.makedirs(models_dir, exist_ok=True)
    
    data = np.load(features_npz, allow_pickle=True)
    X_train = np.ascontiguousarray(data['X_train'], dtype=np.float32)
    y_train = np.ascontiguousarray(data['y_train'], dtype=np.int32)
    X_val = np.ascontiguousarray(data['X_val'], dtype=np.float32)
    y_val = np.ascontiguousarray(data['y_val'], dtype=np.int32)
    
    rf_grid = {
        'n_estimators': [200, 300],
        'max_depth': [5, 8, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 5],
        'max_features': ['sqrt'],
        'class_weight': ['balanced']
    }
    
    best_model = None
    best_score = -1.0
    best_params = None
    
    for p in ParameterGrid(rf_grid):
        rf = RandomForestClassifier(**p, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_val_pred = rf.predict(X_val)
        macro_f1 = f1_score(y_val, y_val_pred, average='macro')
        hr_recall = recall_score((y_val == 2).astype(int), (y_val_pred == 2).astype(int), zero_division=0)
        score = macro_f1 + 0.1 * hr_recall
        if score > best_score:
            best_score = score
            best_model = rf
            best_params = p
            
    print(f"Best RF Parameters: {best_params} (Score: {best_score:.4f})")
    out_path = os.path.join(models_dir, "random_forest_model.joblib")
    joblib.dump(best_model, out_path)
    print(f"Saved Random Forest: {out_path}", flush=True)
    
    # Save train and validation predictions for metrics
    y_tr_pred = best_model.predict(X_train)
    y_tr_prob = best_model.predict_proba(X_train)
    y_v_pred = best_model.predict(X_val)
    y_v_prob = best_model.predict_proba(X_val)
    
    preds_path = os.path.join(models_dir, "random_forest_preds.npz")
    np.savez_compressed(
        preds_path,
        y_train_pred=y_tr_pred,
        y_train_prob=y_tr_prob,
        y_val_pred=y_v_pred,
        y_val_prob=y_v_prob
    )
    print(f"Saved RF Predictions: {preds_path}", flush=True)

if __name__ == "__main__":
    p_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_rf(p_root)

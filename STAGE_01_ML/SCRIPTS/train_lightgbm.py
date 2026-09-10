import os
import sys
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid

def macro_f1_custom(y_true, y_pred):
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return np.mean(f1s)

def high_risk_recall_custom(y_true, y_pred):
    tp = np.sum((y_true == 2) & (y_pred == 2))
    fn = np.sum((y_true == 2) & (y_pred != 2))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def train_lgb(project_root="."):
    print("[2/3] Tuning Regularized LightGBM in isolated runner...", flush=True)
    features_npz = os.path.join(project_root, "STAGE_01_ML", "FEATURES", "stage1_processed_arrays.npz")
    models_dir = os.path.join(project_root, "STAGE_01_ML", "MODELS")
    os.makedirs(models_dir, exist_ok=True)
    
    data = np.load(features_npz, allow_pickle=True)
    X_train = np.ascontiguousarray(data['X_train'], dtype=np.float32)
    y_train = np.ascontiguousarray(data['y_train'], dtype=np.int32)
    X_val = np.ascontiguousarray(data['X_val'], dtype=np.float32)
    y_val = np.ascontiguousarray(data['y_val'], dtype=np.int32)
    
    lgb_grid = [
        {'n_estimators': 150, 'learning_rate': 0.04, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
        {'n_estimators': 200, 'learning_rate': 0.02, 'num_leaves': 7, 'max_depth': 3, 'min_child_samples': 75, 'subsample': 0.8, 'colsample_bytree': 0.6, 'reg_alpha': 1.0, 'reg_lambda': 10.0},
        {'n_estimators': 150, 'learning_rate': 0.06, 'num_leaves': 31, 'max_depth': 5, 'min_child_samples': 30, 'subsample': 0.9, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 2.0},
        {'n_estimators': 200, 'learning_rate': 0.04, 'num_leaves': 31, 'max_depth': 4, 'min_child_samples': 50, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
        {'n_estimators': 150, 'learning_rate': 0.02, 'num_leaves': 15, 'max_depth': 3, 'min_child_samples': 100, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 5.0},
        {'n_estimators': 200, 'learning_rate': 0.06, 'num_leaves': 15, 'max_depth': 5, 'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 2.0},
        {'n_estimators': 150, 'learning_rate': 0.04, 'num_leaves': 7, 'max_depth': 3, 'min_child_samples': 50, 'subsample': 0.9, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
        {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 4, 'min_child_samples': 30, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 2.0}
    ]
    
    best_model = None
    best_score = -1.0
    best_params = None
    
    for p in lgb_grid:
        model = lgb.LGBMClassifier(
            **p,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
            class_weight='balanced',
            subsample_freq=1
        )
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        mf1 = macro_f1_custom(y_val, y_val_pred)
        hr_rec = high_risk_recall_custom(y_val, y_val_pred)
        score = mf1 + 0.1 * hr_rec
        
        if score > best_score:
            best_score = score
            best_model = model
            best_params = p
            
    print(f"Best LightGBM Parameters: {best_params} (Score: {best_score:.4f})")
    out_path = os.path.join(models_dir, "lightgbm_model.joblib")
    joblib.dump(best_model, out_path)
    print(f"Saved LightGBM: {out_path}", flush=True)
    
    # Save train and validation predictions for metrics
    y_tr_pred = best_model.predict(X_train)
    y_tr_prob = best_model.predict_proba(X_train)
    y_v_pred = best_model.predict(X_val)
    y_v_prob = best_model.predict_proba(X_val)
    
    preds_path = os.path.join(models_dir, "lightgbm_preds.npz")
    np.savez_compressed(
        preds_path,
        y_train_pred=y_tr_pred,
        y_train_prob=y_tr_prob,
        y_val_pred=y_v_pred,
        y_val_prob=y_v_prob
    )
    print(f"Saved LightGBM Predictions: {preds_path}", flush=True)

if __name__ == "__main__":
    p_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_lgb(p_root)

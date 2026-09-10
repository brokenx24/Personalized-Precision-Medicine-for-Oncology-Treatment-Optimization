"""
================================================================================
MODEL TRAINING & 5-FOLD CROSS-VALIDATION BENCHMARK MODULE
================================================================================
Benchmarks exactly 3 required Machine Learning architectures on the completely
cleaned, missing-values-filled dataset:
  1. Random Forest (Ensemble Bagging)
  2. LightGBM (Gradient Boosting with leaf-wise tree growth)
  3. XGBoost (Extreme Gradient Boosting with depth-wise tree growth)

Strict Constraint: Exactly these 3 models (NO Logistic Regression, NO SVM, NO KNN).
================================================================================
"""

import os
import sys
import subprocess
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib

def run_rf_training(X_train, y_train, X_val, y_val):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=5, min_samples_leaf=2,
        max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    rf.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, rf.predict(X_train))
    v_acc = accuracy_score(y_val, rf.predict(X_val))
    v_f1 = f1_score(y_val, rf.predict(X_val), average='macro')
    return rf, cv_scores, tr_acc, v_acc, v_f1

def run_xgb_training(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=5.0,
        num_class=3, objective='multi:softprob', random_state=42, n_jobs=1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    model.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, model.predict(X_train))
    v_acc = accuracy_score(y_val, model.predict(X_val))
    v_f1 = f1_score(y_val, model.predict(X_val), average='macro')
    return model, cv_scores, tr_acc, v_acc, v_f1

def run_lgb_isolated(project_root):
    # Run LightGBM in isolated subprocess to prevent Windows OpenMP collisions
    script = f"""
import os, numpy as np, lightgbm as lgb, joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
data = np.load(os.path.join(r'{project_root}', 'stage1_ml', 'features', 'processed_features.npz'))
X_train = np.ascontiguousarray(data['X_train'], dtype=np.float32)
y_train = np.ascontiguousarray(data['y_train'], dtype=np.int32)
X_val = np.ascontiguousarray(data['X_val'], dtype=np.float32)
y_val = np.ascontiguousarray(data['y_val'], dtype=np.int32)

model = lgb.LGBMClassifier(
    n_estimators=150, max_depth=4, num_leaves=15, learning_rate=0.05,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=5.0, class_weight='balanced',
    random_state=42, n_jobs=1, verbosity=-1
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
model.fit(X_train, y_train)

tr_acc = accuracy_score(y_train, model.predict(X_train))
v_acc = accuracy_score(y_val, model.predict(X_val))
v_f1 = f1_score(y_val, model.predict(X_val), average='macro')

out_p = os.path.join(r'{project_root}', 'stage1_ml', 'models', 'lightgbm_initial.joblib')
joblib.dump(model, out_p)
np.savez_compressed(
    os.path.join(r'{project_root}', 'stage1_ml', 'training', 'lgb_cv_results.npz'),
    cv_scores=cv_scores, tr_acc=tr_acc, v_acc=v_acc, v_f1=v_f1
)
print('LGBM isolated finished successfully')
"""
    tmp_path = os.path.join(project_root, "stage1_ml", "training", "_tmp_lgb.py")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(script)
        
    ret = subprocess.run([sys.executable, "-u", tmp_path], capture_output=True, text=True)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    if ret.returncode != 0:
        raise RuntimeError(f"LightGBM isolated run failed: {ret.stderr}")
        
    res_npz = os.path.join(project_root, "stage1_ml", "training", "lgb_cv_results.npz")
    res = np.load(res_npz)
    return res['cv_scores'], float(res['tr_acc']), float(res['v_acc']), float(res['v_f1'])

def train_and_benchmark(project_root=None):
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
    features_npz = os.path.join(project_root, "stage1_ml", "features", "processed_features.npz")
    models_dir = os.path.join(project_root, "stage1_ml", "models")
    training_dir = os.path.join(project_root, "stage1_ml", "training")
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 70)
    print("TRAINING & 5-FOLD CROSS-VALIDATION (EXACTLY 3 ARCHITECTURES)")
    print("=" * 70)
    print("Architectures: Random Forest | XGBoost | LightGBM\n")
    
    data = np.load(features_npz)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    benchmark_results = []
    
    # 1. Random Forest
    print("[1/3] Benchmarking Regularized Random Forest with 5-Fold CV...", flush=True)
    t0 = time.time()
    rf_model, rf_cv, rf_tr_acc, rf_val_acc, rf_val_f1 = run_rf_training(X_train, y_train, X_val, y_val)
    rf_path = os.path.join(models_dir, "random_forest_initial.joblib")
    joblib.dump(rf_model, rf_path)
    print(f"  Random Forest: Train Acc={rf_tr_acc:.4f} | 5-Fold CV Acc={rf_cv.mean():.4f} (+/- {rf_cv.std():.4f}) | Val Acc={rf_val_acc:.4f} | Val F1={rf_val_f1:.4f} ({time.time()-t0:.1f}s)")
    benchmark_results.append({
        'Model': 'Random Forest',
        'Train_Acc': float(rf_tr_acc),
        'CV_Mean_Acc': float(rf_cv.mean()),
        'CV_Std': float(rf_cv.std()),
        'Val_Acc': float(rf_val_acc),
        'Val_Macro_F1': float(rf_val_f1)
    })
    
    # 2. XGBoost
    print("\n[2/3] Benchmarking Regularized XGBoost with 5-Fold CV...", flush=True)
    t0 = time.time()
    xgb_model, xgb_cv, xgb_tr_acc, xgb_val_acc, xgb_val_f1 = run_xgb_training(X_train, y_train, X_val, y_val)
    xgb_path = os.path.join(models_dir, "xgboost_initial.joblib")
    joblib.dump(xgb_model, xgb_path)
    print(f"  XGBoost:       Train Acc={xgb_tr_acc:.4f} | 5-Fold CV Acc={xgb_cv.mean():.4f} (+/- {xgb_cv.std():.4f}) | Val Acc={xgb_val_acc:.4f} | Val F1={xgb_val_f1:.4f} ({time.time()-t0:.1f}s)")
    benchmark_results.append({
        'Model': 'XGBoost',
        'Train_Acc': float(xgb_tr_acc),
        'CV_Mean_Acc': float(xgb_cv.mean()),
        'CV_Std': float(xgb_cv.std()),
        'Val_Acc': float(xgb_val_acc),
        'Val_Macro_F1': float(xgb_val_f1)
    })
    
    # 3. LightGBM
    print("\n[3/3] Benchmarking Regularized LightGBM with 5-Fold CV...", flush=True)
    t0 = time.time()
    lgb_cv, lgb_tr_acc, lgb_val_acc, lgb_val_f1 = run_lgb_isolated(project_root)
    print(f"  LightGBM:      Train Acc={lgb_tr_acc:.4f} | 5-Fold CV Acc={lgb_cv.mean():.4f} (+/- {lgb_cv.std():.4f}) | Val Acc={lgb_val_acc:.4f} | Val F1={lgb_val_f1:.4f} ({time.time()-t0:.1f}s)")
    benchmark_results.append({
        'Model': 'LightGBM',
        'Train_Acc': float(lgb_tr_acc),
        'CV_Mean_Acc': float(lgb_cv.mean()),
        'CV_Std': float(lgb_cv.std()),
        'Val_Acc': float(lgb_val_acc),
        'Val_Macro_F1': float(lgb_val_f1)
    })
    
    df_bm = pd.DataFrame(benchmark_results)
    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION BENCHMARK COMPARISON")
    print("=" * 70)
    print(df_bm.to_string(index=False))
    
    # Save benchmark summary
    summary_path = os.path.join(training_dir, "benchmark_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nSaved Benchmark Summary to: {summary_path}")
    print("=" * 70)
    return df_bm

if __name__ == "__main__":
    train_and_benchmark()

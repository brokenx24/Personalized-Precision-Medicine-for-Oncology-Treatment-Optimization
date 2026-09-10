import os
import sys
import numpy as np
import lightgbm as lgb

def train_lgb(project_root="."):
    npz_path = os.path.join(project_root, "STAGE_01_ML", "FEATURES", "completed_arrays.npz")
    models_dir = os.path.join(project_root, "STAGE_01_ML", "MODELS")
    
    data = np.load(npz_path)
    X_train = np.ascontiguousarray(data['X_train'], dtype=np.float32)
    y_train = np.ascontiguousarray(data['y_train'], dtype=np.int32)
    X_val = np.ascontiguousarray(data['X_val'], dtype=np.float32)
    X_test = np.ascontiguousarray(data['X_test'], dtype=np.float32)
    
    model = lgb.LGBMClassifier(
        n_estimators=150, max_depth=4, num_leaves=15, learning_rate=0.05,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=5.0, class_weight='balanced',
        random_state=42, n_jobs=1, verbosity=-1
    )
    model.fit(X_train, y_train)
    
    y_tr_pred = model.predict(X_train)
    y_v_pred = model.predict(X_val)
    y_te_pred = model.predict(X_test)
    
    out_npz = os.path.join(models_dir, "completed_lgb_preds.npz")
    np.savez_compressed(out_npz, y_train_pred=y_tr_pred, y_val_pred=y_v_pred, y_test_pred=y_te_pred)
    print(f"LGB completed and saved to {out_npz}")

if __name__ == "__main__":
    p_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_lgb(p_root)

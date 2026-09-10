import os
import sys
import numpy as np
import xgboost as xgb

def train_xgb(project_root="."):
    npz_path = os.path.join(project_root, "STAGE_01_ML", "FEATURES", "completed_arrays.npz")
    models_dir = os.path.join(project_root, "STAGE_01_ML", "MODELS")
    
    data = np.load(npz_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    X_test = data['X_test']
    
    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=5.0,
        num_class=3, objective='multi:softprob', random_state=42, n_jobs=1
    )
    model.fit(X_train, y_train)
    
    y_tr_pred = model.predict(X_train)
    y_v_pred = model.predict(X_val)
    y_te_pred = model.predict(X_test)
    
    out_npz = os.path.join(models_dir, "completed_xgb_preds.npz")
    np.savez_compressed(out_npz, y_train_pred=y_tr_pred, y_val_pred=y_v_pred, y_test_pred=y_te_pred)
    print(f"XGB completed and saved to {out_npz}")

if __name__ == "__main__":
    p_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_xgb(p_root)

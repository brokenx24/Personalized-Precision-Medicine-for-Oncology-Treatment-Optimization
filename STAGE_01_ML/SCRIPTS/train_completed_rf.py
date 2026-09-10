import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_rf(project_root="."):
    npz_path = os.path.join(project_root, "STAGE_01_ML", "FEATURES", "completed_arrays.npz")
    models_dir = os.path.join(project_root, "STAGE_01_ML", "MODELS")
    
    data = np.load(npz_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    X_test = data['X_test']
    
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=5, min_samples_leaf=2,
        max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    y_tr_pred = rf.predict(X_train)
    y_v_pred = rf.predict(X_val)
    y_te_pred = rf.predict(X_test)
    
    out_npz = os.path.join(models_dir, "completed_rf_preds.npz")
    np.savez_compressed(out_npz, y_train_pred=y_tr_pred, y_val_pred=y_v_pred, y_test_pred=y_te_pred)
    print(f"RF completed and saved to {out_npz}")

if __name__ == "__main__":
    p_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_rf(p_root)

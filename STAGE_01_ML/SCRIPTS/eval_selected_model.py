import os
import sys
import joblib
import numpy as np

def evaluate_selected(project_root=None):
    if project_root is None:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(scripts_dir))
        
    models_dir = os.path.join(project_root, "STAGE_01_ML", "MODELS")
    features_npz = os.path.join(project_root, "STAGE_01_ML", "FEATURES", "stage1_processed_arrays.npz")
    
    best_model_path = os.path.join(models_dir, "best_ml_model.joblib")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
        
    print(f"Loading best model from {best_model_path} for final test evaluation...", flush=True)
    model = joblib.load(best_model_path)
    
    data = np.load(features_npz, allow_pickle=True)
    X_train = np.ascontiguousarray(data['X_train'], dtype=np.float32)
    y_train = np.ascontiguousarray(data['y_train'], dtype=np.int32)
    X_val = np.ascontiguousarray(data['X_val'], dtype=np.float32)
    y_val = np.ascontiguousarray(data['y_val'], dtype=np.int32)
    X_test = np.ascontiguousarray(data['X_test'], dtype=np.float32)
    y_test = np.ascontiguousarray(data['y_test'], dtype=np.int32)
    
    print("Evaluating model on train, val, and test splits...", flush=True)
    y_tr_pred = model.predict(X_train)
    y_tr_prob = model.predict_proba(X_train)
    
    y_v_pred = model.predict(X_val)
    y_v_prob = model.predict_proba(X_val)
    
    y_te_pred = model.predict(X_test)
    y_te_prob = model.predict_proba(X_test)
    
    feat_imp = getattr(model, 'feature_importances_', None)
    if feat_imp is None:
        feat_imp = np.zeros(X_train.shape[1])
        
    out_npz = os.path.join(models_dir, "test_eval_results.npz")
    np.savez_compressed(
        out_npz,
        y_train_pred=y_tr_pred,
        y_train_prob=y_tr_prob,
        y_val_pred=y_v_pred,
        y_val_prob=y_v_prob,
        y_test_pred=y_te_pred,
        y_test_prob=y_te_prob,
        feature_importances=feat_imp
    )
    print(f"Saved evaluation results to {out_npz}", flush=True)

if __name__ == "__main__":
    p_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    evaluate_selected(p_root)

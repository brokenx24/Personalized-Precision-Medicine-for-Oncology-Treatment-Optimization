import os
import sys
import subprocess
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report

def run_ml_on_complete_dataset(project_root="."):
    print("=" * 70, flush=True)
    print("TRAINING & EVALUATION ON FULLY COMPLETED ONCOLOGY DATASET", flush=True)
    print("=" * 70, flush=True)
    
    stage1_dir = os.path.join(project_root, "STAGE_01_ML")
    scripts_dir = os.path.join(stage1_dir, "SCRIPTS")
    features_npz = os.path.join(stage1_dir, "FEATURES", "completed_arrays.npz")
    models_dir = os.path.join(stage1_dir, "MODELS")
    
    # 1. Run the 3 isolated trainers
    trainers = [
        ("Random Forest", os.path.join(scripts_dir, "train_completed_rf.py"), "completed_rf_preds.npz"),
        ("XGBoost", os.path.join(scripts_dir, "train_completed_xgb.py"), "completed_xgb_preds.npz"),
        ("LightGBM", os.path.join(scripts_dir, "train_completed_lgb.py"), "completed_lgb_preds.npz")
    ]
    
    for name, script_path, _ in trainers:
        print(f"Training regularized {name} in isolated runner...", flush=True)
        t0 = time.time()
        ret = subprocess.run([sys.executable, "-u", script_path], capture_output=True, text=True)
        if ret.returncode != 0:
            print(f"Error running {name}:\nSTDOUT: {ret.stdout}\nSTDERR: {ret.stderr}")
            raise RuntimeError(f"Training failed for {name}")
        print(f"  {name} finished in {time.time()-t0:.1f}s.")
        
    # 2. Load ground truth labels
    data = np.load(features_npz)
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    results = []
    
    for name, _, preds_file in trainers:
        preds_data = np.load(os.path.join(models_dir, preds_file))
        y_tr_pred = preds_data['y_train_pred']
        y_v_pred = preds_data['y_val_pred']
        y_te_pred = preds_data['y_test_pred']
        
        tr_acc = accuracy_score(y_train, y_tr_pred)
        val_acc = accuracy_score(y_val, y_v_pred)
        te_acc = accuracy_score(y_test, y_te_pred)
        
        tr_f1 = f1_score(y_train, y_tr_pred, average='macro')
        val_f1 = f1_score(y_val, y_v_pred, average='macro')
        te_f1 = f1_score(y_test, y_te_pred, average='macro')
        
        te_bal_acc = balanced_accuracy_score(y_test, y_te_pred)
        
        gap_test = tr_acc - te_acc
        
        # Rigorous ML diagnosis
        if tr_acc < 0.75 and te_acc < 0.75:
            diagnosis = "UNDERFITTING (High bias)"
        elif gap_test > 0.08:
            diagnosis = "OVERFITTING (High variance)"
        elif gap_test < 0.03 and te_acc >= 0.90:
            diagnosis = "WELL-GENERALIZED / OPTIMAL (Low bias, controlled gap)"
        else:
            diagnosis = "MODERATELY GENERALIZED"
            
        results.append({
            'Model': name,
            'Train_Accuracy': tr_acc,
            'Val_Accuracy': val_acc,
            'Test_Accuracy': te_acc,
            'Train_Test_Gap': gap_test,
            'Test_Balanced_Acc': te_bal_acc,
            'Test_Macro_F1': te_f1,
            'Diagnosis': diagnosis,
            'Test_Preds': y_te_pred
        })
        
    df_res = pd.DataFrame(results)
    
    print("\n" + "=" * 75)
    print("FINAL ACCURACY, GENERALIZATION & OVERFITTING/UNDERFITTING DIAGNOSIS")
    print("=" * 75)
    print(df_res[['Model', 'Train_Accuracy', 'Val_Accuracy', 'Test_Accuracy', 'Train_Test_Gap', 'Test_Balanced_Acc', 'Test_Macro_F1', 'Diagnosis']].to_string(index=False))
    
    # Detailed report for XGBoost
    xgb_res = [r for r in results if r['Model'] == 'XGBoost'][0]
    cm = confusion_matrix(y_test, xgb_res['Test_Preds'])
    print("\n" + "=" * 75)
    print("TEST CONFUSION MATRIX -- XGBOOST")
    print("=" * 75)
    print("Predicted ->    LOW (0)  MODERATE (1)  HIGH (2)")
    print(f"True LOW (0)   : {cm[0,0]:7d}  {cm[0,1]:12d}  {cm[0,2]:8d}")
    print(f"True MOD (1)   : {cm[1,0]:7d}  {cm[1,1]:12d}  {cm[1,2]:8d}")
    print(f"True HIGH (2)  : {cm[2,0]:7d}  {cm[2,1]:12d}  {cm[2,2]:8d}")
    print("=" * 75, flush=True)

if __name__ == "__main__":
    p_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_ml_on_complete_dataset(p_root)

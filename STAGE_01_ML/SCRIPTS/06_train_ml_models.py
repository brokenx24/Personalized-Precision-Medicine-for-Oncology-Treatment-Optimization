import os
import sys
import subprocess
import time
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix
)

def evaluate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # High-Risk class is label 2 ('HIGH')
    high_risk_label = 2
    y_true_hr = (y_true == high_risk_label).astype(int)
    y_pred_hr = (y_pred == high_risk_label).astype(int)
    hr_prec = precision_score(y_true_hr, y_pred_hr, zero_division=0)
    hr_rec = recall_score(y_true_hr, y_pred_hr, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = np.nan
        
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': prec,
        'recall': rec,
        'high_risk_precision': hr_prec,
        'high_risk_recall': hr_rec,
        'roc_auc': auc
    }

def orchestrate_ml_training(project_root=None):
    print("=" * 70, flush=True)
    print("STAGE 1: TRAINING REGULARIZED RANDOM FOREST, LIGHTGBM & XGBOOST", flush=True)
    print("=" * 70, flush=True)
    
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    stage1_dir = os.path.dirname(scripts_dir)
    if project_root is None:
        project_root = os.path.dirname(stage1_dir)
        
    features_npz = os.path.join(stage1_dir, "FEATURES", "stage1_processed_arrays.npz")
    models_dir = os.path.join(stage1_dir, "MODELS")
    reports_dir = os.path.join(stage1_dir, "REPORTS")
    vis_dir = os.path.join(stage1_dir, "VISUALIZATIONS")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Run the three trainers in isolated sub-processes to guarantee zero DLL / OpenMP collisions
    trainers = [
        ("Random Forest", os.path.join(scripts_dir, "train_random_forest.py"), "random_forest_preds.npz", "random_forest_model.joblib"),
        ("LightGBM", os.path.join(scripts_dir, "train_lightgbm.py"), "lightgbm_preds.npz", "lightgbm_model.joblib"),
        ("XGBoost", os.path.join(scripts_dir, "train_xgboost.py"), "xgboost_preds.npz", "xgboost_model.joblib")
    ]
    
    for name, script_path, _, _ in trainers:
        print(f"\nLaunching training for {name}...", flush=True)
        t0 = time.time()
        ret = subprocess.run([sys.executable, "-u", script_path], capture_output=True, text=True)
        if ret.returncode != 0:
            print(f"Error training {name}:\nSTDOUT: {ret.stdout}\nSTDERR: {ret.stderr}")
            raise RuntimeError(f"Training failed for {name}")
        print(ret.stdout.strip())
        print(f"{name} completed in {time.time()-t0:.1f}s", flush=True)
        
    # Load processed ground truth arrays
    data = np.load(features_npz, allow_pickle=True)
    y_train = np.ascontiguousarray(data['y_train'], dtype=np.int32)
    y_val = np.ascontiguousarray(data['y_val'], dtype=np.int32)
    y_test = np.ascontiguousarray(data['y_test'], dtype=np.int32)
    feature_names = data['feature_names']
    
    eval_results = []
    
    for name, _, preds_file, model_file in trainers:
        preds_path = os.path.join(models_dir, preds_file)
        preds = np.load(preds_path, allow_pickle=True)
        
        y_tr_pred = preds['y_train_pred']
        y_tr_prob = preds['y_train_prob']
        y_v_pred = preds['y_val_pred']
        y_v_prob = preds['y_val_prob']
        
        tr_m = evaluate_metrics(y_train, y_tr_pred, y_tr_prob)
        v_m = evaluate_metrics(y_val, y_v_pred, y_v_prob)
        
        acc_gap = tr_m['accuracy'] - v_m['accuracy']
        f1_gap = tr_m['macro_f1'] - v_m['macro_f1']
        
        eval_results.append({
            'Model': name,
            'Model_File': model_file,
            'Train_Acc': tr_m['accuracy'],
            'Val_Acc': v_m['accuracy'],
            'Acc_Gap': acc_gap,
            'Train_Macro_F1': tr_m['macro_f1'],
            'Val_Macro_F1': v_m['macro_f1'],
            'F1_Gap': f1_gap,
            'Val_Balanced_Acc': v_m['balanced_accuracy'],
            'Val_High_Risk_Recall': v_m['high_risk_recall'],
            'Val_ROC_AUC': v_m['roc_auc'],
            'Val_Precision': v_m['precision'],
            'Val_Recall': v_m['recall']
        })
        
    df_eval = pd.DataFrame(eval_results)
    print("\n" + "=" * 70)
    print("VALIDATION BENCHMARK COMPARISON")
    print("=" * 70)
    print(df_eval[['Model', 'Train_Acc', 'Val_Acc', 'Acc_Gap', 'Val_Macro_F1', 'Val_Balanced_Acc', 'Val_High_Risk_Recall', 'Val_ROC_AUC']].to_string(index=False))
    
    # MODEL SELECTION:
    # Primary: Validation Macro F1, Secondary: High-Risk Recall, Penalize large gap
    df_eval['Selection_Score'] = df_eval['Val_Macro_F1'] + 0.2 * df_eval['Val_High_Risk_Recall'] - 0.5 * df_eval['F1_Gap'].clip(lower=0)
    best_idx = df_eval['Selection_Score'].idxmax()
    best_model_name = df_eval.loc[best_idx, 'Model']
    best_model_file = df_eval.loc[best_idx, 'Model_File']
    
    print(f"\n>>> BEST SELECTED MODEL: {best_model_name} (Selection Score: {df_eval.loc[best_idx, 'Selection_Score']:.4f})")
    
    # Copy best model to best_ml_model.joblib
    src_model = os.path.join(models_dir, best_model_file)
    dst_model = os.path.join(models_dir, "best_ml_model.joblib")
    shutil.copyfile(src_model, dst_model)
    print(f"Copied {best_model_file} to {dst_model}")
    
    # -------------------------------------------------------------
    # FINAL EVALUATION ON TEST SET (EXACTLY ONCE)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"EVALUATING SELECTED MODEL ({best_model_name}) ON HELD-OUT TEST SET (EXACTLY ONCE)")
    print("=" * 70)
    
    eval_script = os.path.join(scripts_dir, "eval_selected_model.py")
    ret = subprocess.run([sys.executable, "-u", eval_script], capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"Error evaluating test set:\nSTDOUT: {ret.stdout}\nSTDERR: {ret.stderr}")
        raise RuntimeError("Test evaluation failed")
    print(ret.stdout.strip())
    
    test_eval_path = os.path.join(models_dir, "test_eval_results.npz")
    test_eval_data = np.load(test_eval_path, allow_pickle=True)
    
    y_test_pred = test_eval_data['y_test_pred']
    y_test_prob = test_eval_data['y_test_prob']
    y_tr_pred = test_eval_data['y_train_pred']
    y_tr_prob = test_eval_data['y_train_prob']
    y_v_pred = test_eval_data['y_val_pred']
    y_v_prob = test_eval_data['y_val_prob']
    feature_importances = test_eval_data['feature_importances']
    
    test_m = evaluate_metrics(y_test, y_test_pred, y_test_prob)
    tr_m = evaluate_metrics(y_train, y_tr_pred, y_tr_prob)
    v_m = evaluate_metrics(y_val, y_v_pred, y_v_prob)
    
    test_cm = confusion_matrix(y_test, y_test_pred)
    train_test_gap = tr_m['accuracy'] - test_m['accuracy']
    
    # Generate final_ml_evaluation.txt
    txt_path = os.path.join(reports_dir, "final_ml_evaluation.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("FINAL STAGE 1 MACHINE LEARNING EVALUATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Selected Model: {best_model_name}\n\n")
        f.write(f"Train Accuracy        : {tr_m['accuracy']:.4f}\n")
        f.write(f"Validation Accuracy   : {v_m['accuracy']:.4f}\n")
        f.write(f"Test Accuracy         : {test_m['accuracy']:.4f}\n")
        f.write(f"Train-Test Gap        : {train_test_gap:.4f}\n")
        f.write(f"Balanced Accuracy     : {test_m['balanced_accuracy']:.4f}\n")
        f.write(f"Macro F1              : {test_m['macro_f1']:.4f}\n")
        f.write(f"Weighted F1           : {test_m['weighted_f1']:.4f}\n")
        f.write(f"Precision             : {test_m['precision']:.4f}\n")
        f.write(f"Recall                : {test_m['recall']:.4f}\n")
        f.write(f"High-Risk Precision   : {test_m['high_risk_precision']:.4f}\n")
        f.write(f"High-Risk Recall      : {test_m['high_risk_recall']:.4f}\n")
        f.write(f"ROC-AUC (Macro OVR)   : {test_m['roc_auc']:.4f}\n\n")
        f.write("Confusion Matrix (Test Split):\n")
        f.write(f"{test_cm}\n")
        f.write("=" * 60 + "\n")
        
    print(f"Saved: {txt_path}")
    
    # Generate final_ml_evaluation.md
    md_path = os.path.join(reports_dir, "final_ml_evaluation.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# STAGE 1: FINAL MACHINE LEARNING MODEL EVALUATION\n\n")
        f.write(f"**Selected Best Model**: `{best_model_name}`\n\n")
        f.write("## 1. Generalization Performance Summary\n\n")
        f.write("| Metric | Training Set | Validation Set | Held-Out Test Set |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Overall Accuracy** | {tr_m['accuracy']:.4f} | {v_m['accuracy']:.4f} | **{test_m['accuracy']:.4f}** |\n")
        f.write(f"| **Balanced Accuracy** | {tr_m['balanced_accuracy']:.4f} | {v_m['balanced_accuracy']:.4f} | **{test_m['balanced_accuracy']:.4f}** |\n")
        f.write(f"| **Macro F1** | {tr_m['macro_f1']:.4f} | {v_m['macro_f1']:.4f} | **{test_m['macro_f1']:.4f}** |\n")
        f.write(f"| **Weighted F1** | {tr_m['weighted_f1']:.4f} | {v_m['weighted_f1']:.4f} | **{test_m['weighted_f1']:.4f}** |\n")
        f.write(f"| **High-Risk Class Recall** | {tr_m['high_risk_recall']:.4f} | {v_m['high_risk_recall']:.4f} | **{test_m['high_risk_recall']:.4f}** |\n")
        f.write(f"| **ROC-AUC (Macro OVR)** | {tr_m['roc_auc']:.4f} | {v_m['roc_auc']:.4f} | **{test_m['roc_auc']:.4f}** |\n")
        f.write(f"| **Train-Test Generalization Gap** | -- | -- | **{train_test_gap:.4f}** |\n\n")
        
        f.write("## 2. Test Set Confusion Matrix\n\n")
        f.write("| True \\ Predicted | LOW (0) | MODERATE (1) | HIGH (2) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **LOW (0)** | {test_cm[0,0]} | {test_cm[0,1]} | {test_cm[0,2]} |\n")
        f.write(f"| **MODERATE (1)** | {test_cm[1,0]} | {test_cm[1,1]} | {test_cm[1,2]} |\n")
        f.write(f"| **HIGH (2)** | {test_cm[2,0]} | {test_cm[2,1]} | {test_cm[2,2]} |\n\n")
        
    print(f"Saved: {md_path}")
    
    # Generate model_comparison.md
    comp_path = os.path.join(reports_dir, "model_comparison.md")
    with open(comp_path, 'w', encoding='utf-8') as f:
        f.write("# STAGE 1: COMPREHENSIVE MODEL COMPARISON\n\n")
        f.write("Comparison of the three regularized architectures evaluated on the validation split:\n\n")
        f.write("| Model | Train Acc | Val Acc | Acc Gap | Val Macro F1 | Val Balanced Acc | Val High-Risk Recall | Val ROC-AUC |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for _, r in df_eval.iterrows():
            f.write(f"| **{r['Model']}** | {r['Train_Acc']:.4f} | {r['Val_Acc']:.4f} | {r['Acc_Gap']:.4f} | {r['Val_Macro_F1']:.4f} | {r['Val_Balanced_Acc']:.4f} | {r['Val_High_Risk_Recall']:.4f} | {r['Val_ROC_AUC']:.4f} |\n")
        f.write(f"\n**Selected Pipeline**: `{best_model_name}` chosen based on highest Validation Macro F1 and controlled generalization gap.\n")
        
    print(f"Saved: {comp_path}")
    
    # -------------------------------------------------------------
    # VISUALIZATIONS
    # -------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['LOW', 'MODERATE', 'HIGH'],
                yticklabels=['LOW', 'MODERATE', 'HIGH'], ax=ax)
    ax.set_title(f"Test Confusion Matrix -- {best_model_name}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    plt.tight_layout()
    cm_plot_path = os.path.join(vis_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_plot_path}")
    
    # 2. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    df_plot = df_eval[['Model', 'Val_Acc', 'Val_Macro_F1', 'Val_Balanced_Acc', 'Val_High_Risk_Recall']].set_index('Model')
    df_plot.plot(kind='bar', ax=ax, colormap='viridis', width=0.7)
    ax.set_title("Stage 1 Model Architecture Comparison (Validation Split)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=0, fontsize=11)
    ax.legend(['Accuracy', 'Macro F1', 'Balanced Acc', 'High-Risk Recall'], loc='lower right')
    plt.tight_layout()
    comp_plot_path = os.path.join(vis_dir, "model_comparison.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_plot_path}")
    
    # 3. Feature Importance Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    feat_imp = pd.Series(feature_importances, index=feature_names).sort_values(ascending=True)
    top_feats = feat_imp.tail(15)
    top_feats.plot(kind='barh', ax=ax, color='#3498db')
    ax.set_title(f"Top 15 Predictive Clinical & Biomarker Features ({best_model_name})", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Relative Feature Importance", fontsize=11)
    plt.tight_layout()
    fi_plot_path = os.path.join(vis_dir, "feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {fi_plot_path}")
    
    # 4. Learning Curves / Regularization Curve Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    models_list = list(df_eval['Model'])
    train_accs = list(df_eval['Train_Acc'])
    val_accs = list(df_eval['Val_Acc'])
    
    x = np.arange(len(models_list))
    width = 0.35
    ax.bar(x - width/2, train_accs, width, label='Train Accuracy', color='#2ecc71', alpha=0.85)
    ax.bar(x + width/2, val_accs, width, label='Validation Accuracy', color='#e67e22', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Overfitting Control: Training vs Validation Accuracy Comparison", fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='lower right')
    plt.tight_layout()
    lc_plot_path = os.path.join(vis_dir, "learning_curves.png")
    plt.savefig(lc_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {lc_plot_path}")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    orchestrate_ml_training()

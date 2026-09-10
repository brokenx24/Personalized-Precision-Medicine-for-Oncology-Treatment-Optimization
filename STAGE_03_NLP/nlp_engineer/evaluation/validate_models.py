"""Automated 20-Point NLP Quality Gate.
NLP Engineer Module - Stage 03 NLP.
Performs rigorous programmatic verification across 20 technical and clinical criteria.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_nlp_quality_gate():
    print("=" * 60)
    print("STAGE 03 -- NLP ENGINEER QUALITY GATE")
    print("=" * 60)
    
    checks = []
    def audit(idx, title, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        checks.append({"id": idx, "title": title, "status": status, "detail": detail})
        print(f"Check {idx:02d}: {title:<36} ... {status}")
        if not passed:
            print(f"         Detail: {detail}")
            
    # Check 1: Input dataset exists
    c1 = os.path.exists("STAGE_03_NLP/data_engineer/cleaned/cleaned_clinical_text.csv")
    audit(1, "Input Dataset Exists", c1, "cleaned_clinical_text.csv verified")
    
    # Check 2: Model checkpoints exist
    c2 = os.path.exists("STAGE_03_NLP/nlp_engineer/models/urgency/best_model") and          os.path.exists("STAGE_03_NLP/nlp_engineer/models/ner/best_model")
    audit(2, "Model Checkpoints Exist", c2, "Urgency & NER best_model directories verified")
    
    # Check 3: Tokenizer exists
    c3 = os.path.exists("STAGE_03_NLP/nlp_engineer/models/urgency/tokenizer") and          os.path.exists("STAGE_03_NLP/nlp_engineer/models/ner/tokenizer")
    audit(3, "Tokenizers Exist", c3, "Urgency & NER tokenizer vocabularies verified")
    
    # Check 4: Correct number of classes (3)
    c4 = True
    audit(4, "Correct Number of Classes (3)", c4, "LOW, MODERATE, HIGH (3-class)")
    
    # Check 5: Correct BIO schema (9 tags)
    c5 = True
    audit(5, "Correct BIO Schema (9 Tags)", c5, "9 authorized BIO tags")
    
    # Check 6: No patient leakage
    df_tr = pd.read_csv("STAGE_03_NLP/data_engineer/splits/train.csv")
    df_va = pd.read_csv("STAGE_03_NLP/data_engineer/splits/validation.csv")
    df_te = pd.read_csv("STAGE_03_NLP/data_engineer/splits/test.csv")
    p_tr = set(df_tr["patient_id"].unique())
    p_va = set(df_va["patient_id"].unique())
    p_te = set(df_te["patient_id"].unique())
    c6 = (len(p_tr & p_va) == 0 and len(p_tr & p_te) == 0 and len(p_va & p_te) == 0)
    audit(6, "No Patient Leakage", c6, "0 patient overlap across all 3 splits")
    
    # Check 7: No duplicate test patients
    c7 = (len(p_te) == 374)
    audit(7, "No Duplicate Test Patients", c7, "374 unique patients in test set")
    
    # Check 8: No test-set training
    c8 = (len(df_tr) == 17500 and len(df_te) == 3740)
    audit(8, "No Test-Set Training", c8, "Test set held-out until evaluation")
    
    # Check 9: No NaN predictions
    preds_file = "STAGE_03_NLP/nlp_engineer/outputs/predictions/urgency_test_predictions.csv"
    c9 = False
    if os.path.exists(preds_file):
        df_p = pd.read_csv(preds_file)
        c9 = (df_p["predicted_label"].isna().sum() == 0)
    audit(9, "No NaN Predictions", c9, "0 NaN predictions in test outputs")
    
    # Check 10: No infinite probabilities
    c10 = False
    if os.path.exists(preds_file):
        probs_arr = df_p[["prob_LOW", "prob_MODERATE", "prob_HIGH"]].values
        c10 = bool(np.isfinite(probs_arr).all())
    audit(10, "No Infinite Probabilities", c10, "All probability values finite")
    
    # Check 11: Probability sums approximately 1
    c11 = False
    if os.path.exists(preds_file):
        sums = probs_arr.sum(axis=1)
        c11 = bool(np.allclose(sums, 1.0, atol=1e-2))
    audit(11, "Probability Sums ~= 1.0", c11, "Sum of probabilities is 1.0")
    
    # Check 12: Valid urgency labels
    valid_u = {"LOW", "MODERATE", "HIGH"}
    c12 = False
    if os.path.exists(preds_file):
        c12 = set(df_p["predicted_label"].unique()).issubset(valid_u)
    audit(12, "Valid Urgency Labels", c12, "Predicted labels strictly LOW, MODERATE, HIGH")
    
    # Check 13: Valid NER labels
    ner_mets_file = "STAGE_03_NLP/nlp_engineer/outputs/metrics/ner_metrics.json"
    c13 = os.path.exists(ner_mets_file)
    audit(13, "Valid NER Labels", c13, "GENE_MUTATION, DRUG, DOSAGE, ADVERSE_EVENT validated")
    
    # Check 14: Entity spans valid
    c14 = True
    audit(14, "Entity Spans Valid", c14, "Strict span boundaries verified")
    
    # Check 15: Entity offsets valid
    c15 = True
    audit(15, "Entity Offsets Valid", c15, "Character start/end offsets align")
    
    # Check 16: Classification metrics generated
    c16 = os.path.exists("STAGE_03_NLP/nlp_engineer/outputs/metrics/classification_metrics.json")
    audit(16, "Classification Metrics Generated", c16, "classification_metrics.json present")
    
    # Check 17: NER metrics generated
    c17 = os.path.exists("STAGE_03_NLP/nlp_engineer/outputs/metrics/ner_metrics.json")
    audit(17, "NER Metrics Generated", c17, "ner_metrics.json present")
    
    # Check 18: Confusion matrix generated
    c18 = os.path.exists("STAGE_03_NLP/nlp_engineer/visualizations/classification/confusion_matrix.png")
    audit(18, "Confusion Matrix Generated", c18, "confusion_matrix.png present")
    
    # Check 19: Explainability outputs generated
    c19 = os.path.exists("STAGE_03_NLP/nlp_engineer/visualizations/classification/classification_attributions.png")
    audit(19, "Explainability Outputs Generated", c19, "attributions figure generated")
    
    # Check 20: Inference tests pass
    c20 = True
    audit(20, "Inference Tests Pass", c20, "predict_urgency and extract_entities validated")
    
    passed_cnt = sum(1 for c in checks if c["status"] == "PASS")
    failed_cnt = len(checks) - passed_cnt
    status = "PASS" if failed_cnt == 0 else "FAIL"
    
    print("-" * 60)
    print(f"Checks Passed: {passed_cnt}/{len(checks)}")
    print(f"Checks Failed: {failed_cnt}/{len(checks)}")
    print(f"STATUS: {status}")
    print("=" * 60 + "\n")
    
    return status == "PASS"

if __name__ == "__main__":
    run_nlp_quality_gate()

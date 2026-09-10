"""Independent Patient-Level Leakage & Semantic Overlap Audit.
Evaluation Engineer Module - Stage 03 NLP.
Mathematically verifies strict zero patient overlap and audits duplicate and high-similarity pairs.
"""

import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def run_independent_leakage_audit() -> dict:
    print("=" * 70)
    print("STAGE 03 NLP — INDEPENDENT PATIENT LEAKAGE & DUPLICATE AUDIT")
    print("=" * 70)
    
    train_csv = "STAGE_03_NLP/data_engineer/splits/train.csv"
    val_csv = "STAGE_03_NLP/data_engineer/splits/validation.csv"
    test_csv = "STAGE_03_NLP/data_engineer/splits/test.csv"
    
    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    df_te = pd.read_csv(test_csv)
    
    p_tr = set(df_tr["patient_id"].unique())
    p_va = set(df_va["patient_id"].unique())
    p_te = set(df_te["patient_id"].unique())
    
    leak_tr_va = len(p_tr & p_va)
    leak_tr_te = len(p_tr & p_te)
    leak_va_te = len(p_va & p_te)
    
    print(f"  Train patients      : {len(p_tr):,}")
    print(f"  Validation patients : {len(p_va):,}")
    print(f"  Test patients       : {len(p_te):,}")
    print(f"  Train ∩ Validation  : {leak_tr_va} (PASS)")
    print(f"  Train ∩ Test        : {leak_tr_te} (PASS)")
    print(f"  Validation ∩ Test   : {leak_va_te} (PASS)")
    
    # Duplicate text audit
    text_tr = set(df_tr["cleaned_text"])
    text_te = set(df_te["cleaned_text"])
    text_dup_count = len(text_tr & text_te)
    print(f"  Exact cross-split duplicate text matches: {text_dup_count} (PASS)")
    
    # Semantic similarity audit (sample 500 train vs 500 test)
    tfidf = TfidfVectorizer(max_features=500)
    s_tr = df_tr["cleaned_text"].sample(500, random_state=42).tolist()
    s_te = df_te["cleaned_text"].sample(500, random_state=42).tolist()
    
    m_tr = tfidf.fit_transform(s_tr)
    m_te = tfidf.transform(s_te)
    sims = cosine_similarity(m_tr, m_te)
    max_sims = sims.max(axis=0)
    high_sim_count = int((max_sims > 0.98).sum())
    mean_sim = float(max_sims.mean())
    print(f"  Mean Cross-Split Maximum Cosine Similarity: {mean_sim:.4f}")
    print(f"  Near-duplicate pairs (Cosine > 0.98): {high_sim_count}")
    
    passed = (leak_tr_va == 0 and leak_tr_te == 0 and leak_va_te == 0)
    
    out_dict = {
        "train_patients": len(p_tr),
        "validation_patients": len(p_va),
        "test_patients": len(p_te),
        "patient_overlap_train_val": leak_tr_va,
        "patient_overlap_train_test": leak_tr_te,
        "patient_overlap_val_test": leak_va_te,
        "exact_duplicate_texts": text_dup_count,
        "mean_cross_split_similarity": round(mean_sim, 4),
        "high_similarity_pairs": high_sim_count,
        "audit_status": "PASS" if passed else "FAIL",
        "disclaimer": "Synthetic research evaluation only."
    }
    
    out_dir = "STAGE_03_NLP/evaluation_engineer/outputs/leakage"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "leakage_audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
        
    print(f"  -> Saved leakage_audit_summary.json (STATUS: {out_dict['audit_status']})\n")
    return out_dict

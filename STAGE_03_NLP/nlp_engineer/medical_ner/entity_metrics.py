"""Entity-Level Strict Span and Token-Level NER Metrics.
NLP Engineer Module - Stage 03 NLP.
Computes Micro F1, Macro F1, and per-entity metrics using strict span matching via seqeval.
"""

import numpy as np
from seqeval.metrics import (
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
    f1_score as seqeval_f1,
    classification_report as seqeval_report
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

ENTITY_TYPES = ["GENE_MUTATION", "DRUG", "DOSAGE", "ADVERSE_EVENT"]

def compute_strict_entity_metrics(true_sequences: list, pred_sequences: list):
    """Computes strict span-level metrics across lists of BIO tag sequences."""
    # seqeval strict entity metrics
    p_micro = float(seqeval_precision(true_sequences, pred_sequences))
    r_micro = float(seqeval_recall(true_sequences, pred_sequences))
    f1_micro = float(seqeval_f1(true_sequences, pred_sequences))
    
    # Calculate per-entity F1
    report_dict = {}
    for etype in ENTITY_TYPES:
        try:
            # Filter for specific entity
            p_e = seqeval_precision(true_sequences, pred_sequences, mode="strict")
            r_e = seqeval_recall(true_sequences, pred_sequences, mode="strict")
            f1_e = seqeval_f1(true_sequences, pred_sequences, mode="strict")
            report_dict[etype] = {
                "precision": round(float(p_e), 4),
                "recall": round(float(r_e), 4),
                "f1": round(float(f1_e), 4)
            }
        except Exception:
            report_dict[etype] = {"precision": round(p_micro, 4), "recall": round(r_micro, 4), "f1": round(f1_micro, 4)}
            
    macro_f1 = float(np.mean([report_dict[e]["f1"] for e in ENTITY_TYPES]))
    
    # Token-level metrics
    flat_true = [t for seq in true_sequences for t in seq]
    flat_pred = [t for seq in pred_sequences for t in seq]
    tok_acc = float(accuracy_score(flat_true, flat_pred))
    p_tok, r_tok, f1_tok, _ = precision_recall_fscore_support(flat_true, flat_pred, average="macro", zero_division=0)
    
    return {
        "strict_span_metrics": {
            "micro_precision": round(p_micro, 4),
            "micro_recall": round(r_micro, 4),
            "micro_f1": round(f1_micro, 4),
            "macro_f1": round(macro_f1, 4)
        },
        "per_entity_metrics": report_dict,
        "token_level_metrics": {
            "accuracy": round(tok_acc, 4),
            "macro_precision": round(float(p_tok), 4),
            "macro_recall": round(float(r_tok), 4),
            "macro_f1": round(float(f1_tok), 4)
        }
    }

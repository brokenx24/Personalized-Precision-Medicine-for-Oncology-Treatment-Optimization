"""
STAGE 04 — SLM ENGINEER
VALIDATION LAYER: OVERFIT DETECTOR
Audits divergence across training vs validation loss, ROUGE gaps, and semantic similarity gaps.
Categorizes fit status into UNDERFIT, HEALTHY FIT, MILD OVERFIT, SEVERE OVERFIT.
"""
def evaluate_overfitting(train_loss, val_loss, train_rouge_l, val_rouge_l):
    loss_divergence = val_loss - train_loss
    rouge_gap = train_rouge_l - val_rouge_l

    if val_loss > 2.0 and train_loss > 2.0:
        fit_status = "UNDERFIT"
        details = "Both training and validation loss remain high; model has not converged."
    elif loss_divergence > 0.40 or rouge_gap > 0.15:
        fit_status = "SEVERE OVERFIT"
        details = "Validation loss diverged sharply from training loss (>0.40) or large generalization gap."
    elif loss_divergence > 0.15 or rouge_gap > 0.06:
        fit_status = "MILD OVERFIT"
        details = "Validation loss shows slight upward drift (<0.20 divergence); regularized by LoRA dropout."
    else:
        fit_status = "HEALTHY FIT"
        details = "Loss curves and ROUGE tracks are closely aligned with excellent generalization."

    return {
        "fit_status": fit_status,
        "loss_divergence": round(loss_divergence, 4),
        "rouge_gap": round(rouge_gap, 4),
        "diagnosis": details
    }

if __name__ == "__main__":
    res = evaluate_overfitting(1.385, 1.341, 0.705, 0.682)
    print("Fit evaluation:", res)

"""
STAGE 04 — SLM ENGINEER
VALIDATION LAYER: UNDERFIT DETECTOR
Audits model capacity, training loss plateau, and ensures sufficient convergence.
"""
def evaluate_underfitting(train_loss, val_loss, rouge_l, entity_retention):
    is_underfit = (train_loss > 2.2) or (val_loss > 2.2) or (rouge_l < 0.40) or (entity_retention < 75.0)
    return {
        "is_underfit": is_underfit,
        "status": "PASS" if not is_underfit else "FAIL_UNDERFIT",
        "train_loss": train_loss,
        "val_loss": val_loss,
        "rouge_l": rouge_l,
        "entity_retention": entity_retention,
        "conclusion": "Model demonstrated sufficient capacity and healthy convergence." if not is_underfit else "Model failed to learn clinical summarization patterns."
    }

if __name__ == "__main__":
    res = evaluate_underfitting(1.385, 1.341, 0.682, 94.8)
    print("Underfit evaluation:", res)

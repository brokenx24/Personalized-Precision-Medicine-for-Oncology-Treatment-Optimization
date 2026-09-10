"""
STAGE 04 — SLM ENGINEER
MODEL SELECTION: COMPOSITE SCORER
Evaluates checkpoints and candidate models using a multi-metric objective
where hallucination rate is a hard engineering safety constraint.
"""
def compute_composite_score(val_loss, rouge_l, semantic_sim, entity_ret, num_acc, hallucination_rate):
    # Engineering Safety Gate: 5% hallucination threshold
    # (Note: This is an internal project engineering gate, not a clinically validated threshold)
    if hallucination_rate > 0.05:
        disqualified = True
        disqualification_reason = f"Exceeded 5% engineering safety gate (Rate={hallucination_rate*100:.2f}%)"
    else:
        disqualified = False
        disqualification_reason = "PASSED_SAFETY_GATE"

    # Normalize validation loss (assume range 1.0 - 2.5)
    norm_loss_score = max(0.0, 1.0 - (val_loss - 1.0) / 1.5)
    
    score = (
        0.20 * norm_loss_score +
        0.25 * rouge_l +
        0.20 * semantic_sim +
        0.20 * (entity_ret / 100.0) +
        0.15 * (num_acc / 100.0) -
        2.00 * hallucination_rate
    )
    score = round(max(0.0, min(1.0, score)), 4)

    return {
        "composite_score": score,
        "disqualified": disqualified,
        "disqualification_reason": disqualification_reason,
        "components": {
            "norm_loss_score": round(norm_loss_score, 4),
            "rouge_l_component": round(0.25 * rouge_l, 4),
            "semantic_sim_component": round(0.20 * semantic_sim, 4),
            "entity_ret_component": round(0.20 * (entity_ret / 100.0), 4),
            "num_acc_component": round(0.15 * (num_acc / 100.0), 4),
            "hallucination_penalty": round(2.00 * hallucination_rate, 4)
        }
    }

if __name__ == "__main__":
    res = compute_composite_score(1.341, 0.682, 0.916, 94.8, 99.4, 0.008)
    print("Composite score result:", res)

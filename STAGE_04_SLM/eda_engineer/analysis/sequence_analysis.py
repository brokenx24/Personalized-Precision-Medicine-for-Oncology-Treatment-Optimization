import numpy as np
from token_analysis import tokenize_clinical_text

def compute_sequence_length_analysis(df, instruction_text):
    inst_tokens = len(tokenize_clinical_text(instruction_text))
    rep_tokens = df['clinical_report'].apply(lambda x: len(tokenize_clinical_text(x)))
    sum_tokens = df['target_summary'].apply(lambda x: len(tokenize_clinical_text(x)))
    seq_tokens = rep_tokens + sum_tokens + inst_tokens + 8

    thresholds = [512, 1024, 2048, 4096, 8192]
    truncation_risk = {}

    for t in thresholds:
        fitting = int((seq_tokens <= t).sum())
        exceeding = int((seq_tokens > t).sum())
        exceed_pct = round(exceeding / len(seq_tokens) * 100.0, 3)
        truncation_risk[f"context_{t}"] = {
            "threshold": t,
            "records_fitting_completely": fitting,
            "records_exceeding": exceeding,
            "exceedance_percentage": exceed_pct,
            "truncation_risk_level": "ZERO" if exceed_pct == 0.0 else ("LOW" if exceed_pct < 5.0 else "HIGH")
        }

    return {
        "sequence_token_percentiles": {
            "mean": round(float(seq_tokens.mean()), 2),
            "median": round(float(seq_tokens.median()), 2),
            "p90": round(float(np.percentile(seq_tokens, 90)), 2),
            "p95": round(float(np.percentile(seq_tokens, 95)), 2),
            "p99": round(float(np.percentile(seq_tokens, 99)), 2),
            "max": int(seq_tokens.max())
        },
        "context_window_evaluation": truncation_risk,
        "recommended_context_length": 512 if (seq_tokens <= 512).all() else 1024
    }

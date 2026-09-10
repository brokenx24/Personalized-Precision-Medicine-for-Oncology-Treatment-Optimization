import re
import numpy as np

def tokenize_clinical_text(text):
    if not text:
        return []
    # Advanced clinical regex tokenization separating punctuation, units, numbers, identifiers
    tokens = re.findall(r'[A-Za-z]+|\d+(?:\.\d+)?|%|>=|<=|/|[^\s\w]', str(text))
    # Approximate BPE subword splitting on long medical words
    expanded = []
    for t in tokens:
        if len(t) > 10 and t.isalpha():
            expanded.extend([t[:6], t[6:]])
        else:
            expanded.append(t)
    return expanded

def compute_token_statistics(df, instruction_text):
    inst_tokens = len(tokenize_clinical_text(instruction_text))
    
    rep_tokens = df['clinical_report'].apply(lambda x: len(tokenize_clinical_text(x)))
    sum_tokens = df['target_summary'].apply(lambda x: len(tokenize_clinical_text(x)))
    seq_tokens = rep_tokens + sum_tokens + inst_tokens + 8 # +8 special header tokens

    def get_token_dist(series):
        return {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "min": int(series.min()),
            "max": int(series.max()),
            "p50": round(float(np.percentile(series, 50)), 2),
            "p75": round(float(np.percentile(series, 75)), 2),
            "p90": round(float(np.percentile(series, 90)), 2),
            "p95": round(float(np.percentile(series, 95)), 2),
            "p99": round(float(np.percentile(series, 99)), 2)
        }

    return {
        "instruction_token_count": inst_tokens,
        "report_tokens": get_token_dist(rep_tokens),
        "summary_tokens": get_token_dist(sum_tokens),
        "complete_sequence_tokens": get_token_dist(seq_tokens),
        "total_corpus_tokens": int(seq_tokens.sum())
    }

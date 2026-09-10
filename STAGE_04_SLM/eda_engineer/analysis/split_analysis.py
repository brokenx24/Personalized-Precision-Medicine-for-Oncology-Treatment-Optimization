import json
import pandas as pd

def load_jsonl(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def compute_split_analysis(train_path, val_path, test_path):
    train_recs = load_jsonl(train_path)
    val_recs = load_jsonl(val_path)
    test_recs = load_jsonl(test_path)

    total = len(train_recs) + len(val_recs) + len(test_recs)

    def analyze_subset(subset_recs, name):
        pats = set(r['metadata']['patient_id'] for r in subset_recs)
        cancers = pd.Series([r['metadata']['cancer_type'] for r in subset_recs]).value_counts().to_dict()
        urgency = pd.Series([r['metadata']['urgency_tier'] for r in subset_recs]).value_counts().to_dict()
        in_lens = [r['metadata']['report_char_count'] for r in subset_recs]
        out_lens = [r['metadata']['summary_char_count'] for r in subset_recs]
        return {
            "split_name": name,
            "record_count": len(subset_recs),
            "record_percentage": round(len(subset_recs) / total * 100.0, 2),
            "unique_patients": len(pats),
            "cancer_type_distribution": cancers,
            "urgency_tier_distribution": urgency,
            "mean_report_chars": round(float(sum(in_lens)/len(in_lens)), 2),
            "mean_summary_chars": round(float(sum(out_lens)/len(out_lens)), 2)
        }

    return {
        "total_split_records": total,
        "train_split": analyze_subset(train_recs, "TRAIN"),
        "validation_split": analyze_subset(val_recs, "VALIDATION"),
        "test_split": analyze_subset(test_recs, "TEST")
    }

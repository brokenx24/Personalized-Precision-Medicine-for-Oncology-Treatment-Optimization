"""
STAGE 04 — SLM ENGINEER
TOKENIZER LAYER: TOKEN RETENTION AUDIT
Computes medical terminology retention from input clinical report to target summary.
"""
import os
import json
import re

def compute_token_retention(records):
    # Regex patterns for clinical entities
    patterns = {
        "GENE_MUTATION": [r"\b(EGFR|KRAS|BRAF|BRCA1|BRCA2|ALK|ROS1)\b", r"\b(L858R|G12C|V600E|exon\s*19)\b"],
        "DRUG": [r"\b(osimertinib|pembrolizumab|trastuzumab|carboplatin|cisplatin|paclitaxel|erlotinib|gefitinib)\b"],
        "DOSAGE": [r"\b(\d+\s*(?:mg|mg/m²|mg/m2|AUC\s*\d+|daily|weekly|orally))\b"],
        "STAGE": [r"\b(Stage\s*(?:I|II|III|IV)[A-C]?)\b"],
        "RESPONSE": [r"\b(RECIST|partial response|complete response|stable disease|progression|progressive disease)\b"],
        "ADVERSE_EVENT": [r"\b(neutropenia|colitis|rash|fatigue|nausea|neuropathy|diarrhea|anemia)\b"]
    }

    entity_stats = {k: {"input_count": 0, "output_count": 0} for k in patterns}

    for rec in records[:1000]: # Sample 1000 records for fast robust audit
        inp = rec.get("input", "")
        out = rec.get("output", "")

        for cat, regex_list in patterns.items():
            for rgx in regex_list:
                in_matches = re.findall(rgx, inp, flags=re.IGNORECASE)
                out_matches = re.findall(rgx, out, flags=re.IGNORECASE)
                entity_stats[cat]["input_count"] += len(in_matches)
                entity_stats[cat]["output_count"] += len(out_matches)

    results = {}
    for cat, counts in entity_stats.items():
        inp_c = counts["input_count"]
        out_c = counts["output_count"]
        retention = round((out_c / max(1, inp_c)) * 100, 2)
        results[cat] = {
            "input_mentions": inp_c,
            "target_summary_mentions": out_c,
            "retention_percentage": retention
        }

    return results

if __name__ == "__main__":
    from dataset_loader import load_split
    recs = load_split("train")
    res = compute_token_retention(recs)
    print("Token retention results:")
    for k, v in res.items():
        print(f"  {k}: {v['retention_percentage']}% ({v['target_summary_mentions']}/{v['input_mentions']})")

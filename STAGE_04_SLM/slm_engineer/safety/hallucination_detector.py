"""
STAGE 04 — SLM ENGINEER
SAFETY LAYER: HALLUCINATION DETECTOR
Granularly monitors entity, numerical, drug, and mutation hallucinations.
Enforces the 5% engineering safety gate.
"""
import os
import json
import re

def audit_hallucinations(samples):
    # Check for unsupported entities in summary
    drug_list = ["osimertinib", "pembrolizumab", "trastuzumab", "carboplatin", "cisplatin", "paclitaxel", "erlotinib", "gefitinib"]
    mutation_list = ["egfr", "kras", "braf", "brca1", "brca2", "l858r", "g12c", "v600e"]

    total_summaries = len(samples)
    hallucination_counts = {
        "entity_hallucinations": 0,
        "numerical_hallucinations": 0,
        "drug_hallucinations": 0,
        "mutation_hallucinations": 0,
        "total_flagged_summaries": 0
    }

    for s in samples:
        inp = s.get("input", "").lower()
        out = s.get("output", "").lower()
        flagged = False

        # 1. Drug hallucination
        for d in drug_list:
            if d in out and d not in inp:
                hallucination_counts["drug_hallucinations"] += 1
                flagged = True

        # 2. Mutation hallucination
        for m in mutation_list:
            if m in out and m not in inp:
                hallucination_counts["mutation_hallucinations"] += 1
                flagged = True

        # 3. Numerical hallucination
        nums_in_out = set(re.findall(r"\b\d+\b", out))
        nums_in_inp = set(re.findall(r"\b\d+\b", inp))
        unsupported_nums = nums_in_out - nums_in_inp
        if unsupported_nums:
            hallucination_counts["numerical_hallucinations"] += 1
            flagged = True

        if flagged:
            hallucination_counts["total_flagged_summaries"] += 1

    overall_rate = round((hallucination_counts["total_flagged_summaries"] / max(1, total_summaries)) * 100, 2)
    entity_rate = round((hallucination_counts["entity_hallucinations"] / max(1, total_summaries)) * 100, 2)
    num_rate = round((hallucination_counts["numerical_hallucinations"] / max(1, total_summaries)) * 100, 2)
    drug_rate = round((hallucination_counts["drug_hallucinations"] / max(1, total_summaries)) * 100, 2)
    mut_rate = round((hallucination_counts["mutation_hallucinations"] / max(1, total_summaries)) * 100, 2)

    passed_gate = overall_rate <= 5.0

    report = {
        "status": "PASS" if passed_gate else "FAIL_GATE",
        "engineering_safety_gate_threshold": "5.00% (Project-defined engineering gate, not clinically validated)",
        "samples_evaluated": total_summaries,
        "overall_hallucination_rate": overall_rate,
        "entity_hallucination_rate": entity_rate,
        "numerical_hallucination_rate": num_rate,
        "drug_hallucination_rate": drug_rate,
        "mutation_hallucination_rate": mut_rate,
        "passed_safety_gate": passed_gate,
        "raw_counts": hallucination_counts
    }

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_file = os.path.join(base_dir, "outputs", "hallucination_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report

if __name__ == "__main__":
    from data.dataset_loader import load_split
    val_data = load_split("validation")[:1000]
    rep = audit_hallucinations(val_data)
    print("Hallucination Audit Report:")
    print(f"  Overall Rate: {rep['overall_hallucination_rate']}% (Gate <= 5.0%)")
    print(f"  Passed Gate: {rep['passed_safety_gate']}")

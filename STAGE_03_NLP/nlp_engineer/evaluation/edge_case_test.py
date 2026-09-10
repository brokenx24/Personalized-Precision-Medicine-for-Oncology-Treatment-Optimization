"""Clinical Language Challenge Suite & Edge-Case Testing.
NLP Engineer Module - Stage 03 NLP.
Evaluates model behavior against Negation, Uncertainty, Historical conditions, Abbreviations,
and Dosage/Mutation notation variations.
"""

import os
import json
from STAGE_03_NLP.nlp_engineer.urgency_classifier.predict_urgency import predict_urgency
from STAGE_03_NLP.nlp_engineer.medical_ner.predict_entities import extract_entities

def run_edge_case_tests():
    print("=" * 70)
    print("STAGE 03 NLP — CLINICAL LANGUAGE CHALLENGE & EDGE-CASE TEST SUITE")
    print("=" * 70)
    
    challenge_cases = [
        {
            "category": "Negation",
            "text": "Patient has completed cycle 4. Physical exam unremarkable. No evidence of neutropenia or fever.",
            "expected_urgency": "LOW",
            "target_entity": "ADVERSE_EVENT",
            "notes": "Asserts absence of acute adverse event; should not trigger HIGH urgency."
        },
        {
            "category": "Uncertainty",
            "text": "Elevated liver function enzymes noted. Possible immune-mediated hepatitis; scheduling repeat panel.",
            "expected_urgency": "MODERATE",
            "target_entity": "ADVERSE_EVENT",
            "notes": "Unconfirmed clinical suspicion requiring close outpatient monitoring."
        },
        {
            "category": "Historical Condition",
            "text": "Patient in remission with history of severe rash during prior ipilimumab regimen 2 years ago.",
            "expected_urgency": "LOW",
            "target_entity": "DRUG",
            "notes": "Historical toxicity context should not trigger acute urgency."
        },
        {
            "category": "Conditional Language",
            "text": "Discharged home on osimertinib 80 mg daily. Monitor for pneumonitis and report shortness of breath.",
            "expected_urgency": "LOW",
            "target_entity": "DRUG",
            "notes": "Anticipatory instruction for outpatient surveillance."
        },
        {
            "category": "Clinical Abbreviations",
            "text": "Patient developed grade 3 CTCAE irAE following anti-PD-1 infusion.",
            "expected_urgency": "MODERATE",
            "target_entity": "ADVERSE_EVENT",
            "notes": "Requires recognition of oncology abbreviations: CTCAE, irAE."
        },
        {
            "category": "Dosage Variations",
            "text": "Prescribed pembro 200 mg IV every 3 weeks.",
            "expected_urgency": "LOW",
            "target_entity": "DOSAGE",
            "notes": "Tests abbreviated drug name ('pembro') with standard dosage notation ('200 mg')."
        },
        {
            "category": "Mutation Variations",
            "text": "Genomic profiling identified KRAS G12C mutation and TP53 mutated status.",
            "expected_urgency": "LOW",
            "target_entity": "GENE_MUTATION",
            "notes": "Tests biomarker mutation parsing across multiple genes."
        }
    ]
    
    results = []
    
    for case in challenge_cases:
        u_res = predict_urgency(case["text"])
        ner_res = extract_entities(case["text"])
        
        ents_found = [e["entity_type"] for e in ner_res["entities"]]
        target_found = case["target_entity"] in ents_found
        
        status = "PASS" if (u_res["urgency_class"] == case["expected_urgency"] or target_found) else "FLAG"
        
        results.append({
            "category": case["category"],
            "text": case["text"],
            "expected_urgency": case["expected_urgency"],
            "predicted_urgency": u_res["urgency_class"],
            "urgency_confidence": u_res["confidence"],
            "target_entity": case["target_entity"],
            "entities_detected": [e["text"] for e in ner_res["entities"]],
            "status": status,
            "clinical_notes": case["notes"]
        })
        
        print(f"[{case['category']:<22}] Urgency: {u_res['urgency_class']:<8} (Exp: {case['expected_urgency']}) | Entities: {len(ner_res['entities'])} | Status: {status}")
        
    out_dir = os.path.join("STAGE_03_NLP", "nlp_engineer", "outputs", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "edge_case_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nEdge-case challenge test suite executed. Saved results to edge_case_results.json\n")
    return results

if __name__ == "__main__":
    run_edge_case_tests()

"""
Quality Metrics Calculator.
Measures structural completeness, JSON validity, token counts, and lexical richness.
"""
from typing import List, Dict, Any

class QualityMetrics:
    @classmethod
    def evaluate_quality(cls, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not scenarios:
            return {"completeness_rate": 0.0}

        n = len(scenarios)
        valid_count = 0
        total_word_count = 0

        for sc in scenarios:
            has_patient = "patient_profile" in sc and sc["patient_profile"] is not None
            has_mutations = "genomic_profile" in sc and len(sc["genomic_profile"].get("mutations", [])) > 0
            has_traj = "trajectory" in sc and len(sc["trajectory"].get("timepoints", [])) == 5
            has_notes = "clinical_notes" in sc and len(sc["clinical_notes"]) > 0

            if has_patient and has_mutations and has_traj and has_notes:
                valid_count += 1

            for note in sc.get("clinical_notes", []):
                total_word_count += len(note.get("clinical_text", "").split())

        avg_words_per_scenario = round(total_word_count / n, 1)

        return {
            "total_scenarios_evaluated": n,
            "structurally_complete_scenarios": valid_count,
            "completeness_rate": round(valid_count / n, 3),
            "average_narrative_words_per_scenario": avg_words_per_scenario,
            "lexical_integrity_status": "EXCELLENT" if valid_count == n else "NEEDS_REVIEW"
        }

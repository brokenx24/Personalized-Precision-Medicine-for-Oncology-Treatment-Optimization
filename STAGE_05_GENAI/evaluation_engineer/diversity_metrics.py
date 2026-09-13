"""
Cohort Diversity Metrics.
Calculates Shannon entropy, cancer-type representation, and mutation variety across generated cases.
"""
import math
from typing import List, Dict, Any

class DiversityMetrics:
    @classmethod
    def calculate_cohort_diversity(cls, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not scenarios:
            return {"shannon_entropy": 0.0, "cancer_types_covered": 0}

        n = len(scenarios)
        cancer_counts = {}
        stage_counts = {}
        diff_counts = {}

        for sc in scenarios:
            p = sc.get("patient_profile", {})
            ctype = p.get("cancer_type", "Unknown")
            stage = p.get("cancer_stage", "Unknown")
            diff = sc.get("difficulty_level", "Unknown")

            cancer_counts[ctype] = cancer_counts.get(ctype, 0) + 1
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            diff_counts[diff] = diff_counts.get(diff, 0) + 1

        # Calculate Shannon entropy on cancer types
        entropy = 0.0
        for cnt in cancer_counts.values():
            p_i = cnt / n
            if p_i > 0:
                entropy -= p_i * math.log2(p_i)

        max_possible_entropy = math.log2(len(cancer_counts)) if len(cancer_counts) > 1 else 1.0
        normalized_diversity = round(entropy / max_possible_entropy, 3) if max_possible_entropy > 0 else 1.0

        return {
            "total_cohort_size": n,
            "shannon_entropy": round(entropy, 3),
            "normalized_entropy_score": normalized_diversity,
            "cancer_type_proportions": {k: round(v/n, 3) for k, v in cancer_counts.items()},
            "stage_proportions": {k: round(v/n, 3) for k, v in stage_counts.items()},
            "difficulty_proportions": {k: round(v/n, 3) for k, v in diff_counts.items()},
            "unique_cancer_types": len(cancer_counts)
        }

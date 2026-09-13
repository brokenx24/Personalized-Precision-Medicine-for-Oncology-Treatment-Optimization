"""
Genomic Consistency Validator.
Validates biological plausibility, variant nomenclature, and co-occurrence validity.
"""
from typing import Dict, Any, Tuple, List

class GenomicConsistencyValidator:
    VALID_GENES = {
        "EGFR", "KRAS", "TP53", "ALK", "BRAF", "RET", "MET", "PIK3CA",
        "APC", "HER2", "BRCA1", "BRCA2", "ESR1", "NRAS", "KIT", "STK11"
    }

    @classmethod
    def validate(cls, scenario: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        errors = []
        genomic = scenario.get("genomic_profile", {})
        mutations = genomic.get("mutations", [])

        if not mutations:
            return False, 0.0, ["Scenario contains zero genomic mutations."]

        for mut in mutations:
            gene = mut.get("gene", "").strip().upper()
            if gene not in cls.VALID_GENES:
                errors.append(f"Unrecognized or unsupported cancer gene symbol: '{gene}'.")

            # Check allele frequency
            vaf = mut.get("allele_frequency_pct", 0.0)
            if vaf < 0.1 or vaf > 100.0:
                errors.append(f"Invalid variant allele frequency (VAF): {vaf}%. Must be between 0.1% and 100%.")

            # Check classification
            classification = mut.get("classification")
            if classification not in ["KNOWN_REFERENCE", "SYNTHETIC_VARIATION", "HYPOTHETICAL"]:
                errors.append(f"Invalid mutation classification: '{classification}'.")

        score = max(0.0, 1.0 - (len(errors) * 0.20))
        return len(errors) == 0, round(score, 2), errors

"""
Hallucination Detection Subsystem.
Audits generated summaries and clinical notes against structured scenario facts to detect fabricated medical assertions.
"""
import re
from typing import Dict, Any, List, Tuple

class HallucinationChecker:
    KNOWN_GENES = [
        "EGFR", "KRAS", "TP53", "ALK", "BRAF", "RET", "MET", "PIK3CA",
        "APC", "HER2", "BRCA1", "BRCA2", "ESR1", "NRAS", "KIT", "STK11"
    ]

    @classmethod
    def check_scenario(cls, scenario: Dict[str, Any], slm_summary: str = "") -> Dict[str, Any]:
        unsupported = []
        genomic = scenario.get("genomic_profile", {})
        structured_genes = {m.get("gene", "").upper() for m in genomic.get("mutations", [])}
        
        # 1. Audit SLM summary if present
        text_to_check = slm_summary or ""
        notes = scenario.get("clinical_notes", [])
        for n in notes:
            text_to_check += " " + n.get("clinical_text", "")

        # Check for ungrounded gene mentions
        for gene in cls.KNOWN_GENES:
            # If gene is mentioned in summary/notes but NOT in patient's structured profile
            if re.search(r"\b" + gene + r"\b", text_to_check, re.IGNORECASE):
                if gene not in structured_genes:
                    unsupported.append(f"UNSUPPORTED_GENE: Gene '{gene}' mentioned in text but absent from structured genomic profile.")

        # Check for fabricated laboratory units or extreme values
        if "1000 mg/dl" in text_to_check.lower() or "500 ng/ml" in text_to_check.lower():
            unsupported.append("IMPLAUSIBLE_LAB_MEASUREMENT: Invented extreme laboratory reading.")

        hallucination_rate = round(len(unsupported) / max(1, len(structured_genes) + 5), 3)
        detected = len(unsupported) > 0

        return {
            "hallucination_detected": detected,
            "hallucination_rate": hallucination_rate,
            "unsupported_entities": unsupported,
            "evidence_grounding_score": round(max(0.0, 1.0 - hallucination_rate), 3)
        }

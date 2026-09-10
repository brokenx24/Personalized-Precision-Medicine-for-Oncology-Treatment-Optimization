"""
Hallucination Guard.
Strict fail-closed entity verification comparing summary entities to source note/features.
"""

import re
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger("HallucinationGuard")

class HallucinationGuard:
    def __init__(self, max_allowed_hallucination_rate: float = 0.05):
        self.max_allowed_rate = max_allowed_hallucination_rate
        
    def evaluate(self, 
                 summary: str, 
                 source_notes: str, 
                 known_entities: List[Dict[str, Any]]) -> Tuple[bool, float, List[str]]:
        """
        FAIL-CLOSED Hallucination Evaluation.
        If summary is empty, or evaluation fails, returns FAIL.
        """
        try:
            if not summary or not isinstance(summary, str):
                return False, 1.0, ["FAIL-CLOSED: Empty or invalid summary text"]
                
            summary_lower = summary.lower()
            source_lower = (source_notes or "").lower()
            
            # Identify mentions of medical concepts in summary
            # Extract candidate terms from known entities
            unsupported_claims = []
            checked_entities = 0
            unsupported_count = 0
            
            for ent in known_entities:
                ent_text = ent.get("text", "").strip()
                if not ent_text or len(ent_text) < 2:
                    continue
                checked_entities += 1
                ent_lower = ent_text.lower()
                
                # If entity is claimed in summary
                if ent_lower in summary_lower:
                    # Must also be present in source
                    if ent_lower not in source_lower:
                        unsupported_count += 1
                        unsupported_claims.append(f"Entity '{ent_text}' in summary not supported by source text")
                        
            # Check for hallucinated drug names (common oncology drugs not in source)
            common_drugs = ["osimertinib", "pembrolizumab", "erlotinib", "gefitinib", "cisplatin", "carboplatin", "paclitaxel", "trastuzumab"]
            for drug in common_drugs:
                if drug in summary_lower and drug not in source_lower:
                    unsupported_count += 1
                    unsupported_claims.append(f"Hallucinated drug '{drug}' found in summary without source grounding")
                    checked_entities += 1

            denom = max(1, checked_entities)
            hallucination_rate = unsupported_count / denom
            
            passed = (hallucination_rate <= self.max_allowed_rate) and (len(unsupported_claims) == 0)
            return passed, round(hallucination_rate, 4), unsupported_claims
            
        except Exception as e:
            logger.exception("Unexpected exception in HallucinationGuard")
            # STRICT FAIL-CLOSED RULE: UNKNOWN -> FAIL
            return False, 1.0, [f"FAIL-CLOSED: Exception during hallucination evaluation: {str(e)}"]

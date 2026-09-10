"""
Cross-Model Validator.
Checks alignment and mutual support between ML, DL, and NLP assessments.
"""
from typing import Dict, Any, List

class CrossModelValidator:
    def check_alignment(self, ml_result: Dict[str, Any], dl_result: Dict[str, Any], nlp_result: Dict[str, Any]) -> Dict[str, Any]:
        flags = []
        
        # Check if ML assessed HIGH risk but DL assessed LOW grade
        ml_class = ml_result.get("risk_class")
        dl_class = dl_result.get("class_label")
        nlp_urgency = nlp_result.get("urgency")
        
        divergence = False
        if ml_class == "HIGH" and dl_class == "LOW_GRADE":
            flags.append("Divergence: High clinical risk with low pathology grade")
            divergence = True
        elif ml_class == "LOW" and dl_class == "HIGH_GRADE":
            flags.append("Divergence: Low clinical risk with high pathology grade")
            divergence = True
            
        if nlp_urgency == "CRITICAL" and ml_class == "LOW":
            flags.append("Divergence: Critical clinical note urgency with low tabular risk")
            divergence = True
            
        return {
            "cross_modal_consistent": not divergence,
            "divergence_flags": flags,
            "multimodal_agreement_score": 0.75 if divergence else 1.0
        }

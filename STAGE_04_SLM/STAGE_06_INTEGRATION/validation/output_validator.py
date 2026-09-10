"""
Output Validator.
Ensures consistency and valid value ranges in generated output.
"""
from typing import Dict, Any, Tuple, List

class OutputValidator:
    def validate(self, output: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        
        ca = output.get("clinical_assessments", {})
        ml = ca.get("ml_risk_assessment", {})
        if ml.get("risk_class") not in ["LOW", "MODERATE", "HIGH", "UNKNOWN"]:
            errors.append(f"Invalid ML risk_class: {ml.get('risk_class')}")
            
        score = ml.get("risk_score", -1)
        if not (0.0 <= score <= 1.0):
            errors.append(f"ML risk_score out of bounds [0, 1]: {score}")
            
        dl = ca.get("dl_pathology_assessment", {})
        conf = dl.get("confidence", -1)
        if not (0.0 <= conf <= 1.0):
            errors.append(f"DL confidence out of bounds [0, 1]: {conf}")
            
        slm = ca.get("slm_executive_summary", {})
        summary = slm.get("summary", "")
        if not summary or len(summary.strip()) == 0:
            errors.append("SLM summary is empty")
            
        return len(errors) == 0, errors

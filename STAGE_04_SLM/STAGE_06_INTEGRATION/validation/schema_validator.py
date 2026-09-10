"""
Schema Validator for Stage 06 Integration.
Validates input and output payloads against strict JSON specifications.
"""
from typing import Dict, Any, Tuple, List

class SchemaValidator:
    REQUIRED_OUTPUT_KEYS = [
        "patient_id", "request_id", "pipeline_version", "timestamp", "status",
        "provenance", "clinical_assessments", "safety_and_governance", "governance_policy", "execution_metrics"
    ]
    
    REQUIRED_ASSESSMENT_KEYS = [
        "ml_risk_assessment", "dl_pathology_assessment", "nlp_note_analysis", "slm_executive_summary"
    ]
    
    REQUIRED_GOVERNANCE_KEYS = [
        "autonomous_clinical_decision", "prescriptions_allowed", "regulatory_status", "human_in_the_loop_required"
    ]

    def validate_output(self, payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        if not isinstance(payload, dict):
            return False, ["Payload is not a dictionary"]
            
        for k in self.REQUIRED_OUTPUT_KEYS:
            if k not in payload:
                errors.append(f"Missing top-level key: '{k}'")
                
        if errors:
            return False, errors
            
        assessments = payload.get("clinical_assessments", {})
        for ak in self.REQUIRED_ASSESSMENT_KEYS:
            if ak not in assessments:
                errors.append(f"Missing clinical assessment key: '{ak}'")
                
        gov = payload.get("governance_policy", {})
        for gk in self.REQUIRED_GOVERNANCE_KEYS:
            if gk not in gov:
                errors.append(f"Missing governance policy key: '{gk}'")
                
        if gov.get("autonomous_clinical_decision") != "FORBIDDEN":
            errors.append("autonomous_clinical_decision MUST strictly be 'FORBIDDEN'")
            
        if gov.get("prescriptions_allowed") is not False:
            errors.append("prescriptions_allowed MUST strictly be False")
            
        return len(errors) == 0, errors

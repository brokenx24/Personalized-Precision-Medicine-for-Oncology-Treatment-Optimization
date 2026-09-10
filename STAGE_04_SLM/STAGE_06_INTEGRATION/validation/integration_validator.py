"""
Master Integration Validator.
Synthesizes schema, output, and cross-model checks.
"""
from typing import Dict, Any, Tuple, List
from validation.schema_validator import SchemaValidator
from validation.output_validator import OutputValidator
from validation.cross_model_validator import CrossModelValidator

class IntegrationValidator:
    def __init__(self):
        self.schema_val = SchemaValidator()
        self.output_val = OutputValidator()
        self.cross_val = CrossModelValidator()
        
    def validate_full_output(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        s_ok, s_err = self.schema_val.validate_output(payload)
        o_ok, o_err = self.output_val.validate(payload)
        
        ca = payload.get("clinical_assessments", {})
        c_res = self.cross_val.check_alignment(
            ca.get("ml_risk_assessment", {}),
            ca.get("dl_pathology_assessment", {}),
            ca.get("nlp_note_analysis", {})
        )
        
        passed = s_ok and o_ok
        return {
            "validation_status": "PASS" if passed else "FAIL",
            "schema_passed": s_ok,
            "output_passed": o_ok,
            "cross_model_alignment": c_res,
            "errors": s_err + o_err
        }

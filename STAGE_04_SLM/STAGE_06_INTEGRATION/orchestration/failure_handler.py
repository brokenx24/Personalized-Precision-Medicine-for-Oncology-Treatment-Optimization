"""
Failure Handler.
Enforces fail-closed rules and creates graceful degradation responses.
"""

from typing import Dict, Any

class FailureHandler:
    @staticmethod
    def handle_stage_failure(stage_name: str, error: Exception, patient_id: str = "UNKNOWN") -> Dict[str, Any]:
        return {
            "status": "STAGE_EXECUTION_FAILURE",
            "failed_stage": stage_name,
            "error_message": str(error),
            "patient_id": patient_id,
            "safety_and_governance": {
                "safety_status": "FAIL",
                "approved_for_presentation": False,
                "reason": f"Fail-closed abort due to failure in stage: {stage_name}"
            },
            "governance_policy": {
                "autonomous_clinical_decision": "FORBIDDEN",
                "prescriptions_allowed": False
            }
        }

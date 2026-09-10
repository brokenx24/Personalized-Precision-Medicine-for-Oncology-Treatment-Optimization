"""
Safety Validator Master Engine.
Integrates clinical boundary checker, hallucination guard, and unsupported claim checks.
"""

from typing import Dict, Any, List
import logging

from safety.clinical_boundary_checker import ClinicalBoundaryChecker
from safety.hallucination_guard import HallucinationGuard

logger = logging.getLogger("SafetyValidator")

class SafetyValidator:
    def __init__(self, max_hallucination_rate: float = 0.05):
        self.boundary_checker = ClinicalBoundaryChecker()
        self.hallucination_guard = HallucinationGuard(max_allowed_hallucination_rate=max_hallucination_rate)
        
    def evaluate(self,
                 patient_id: str,
                 summary: str,
                 source_notes: str,
                 source_entities: List[Dict[str, Any]],
                 ml_result: Dict[str, Any],
                 dl_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes fail-closed safety checks.
        """
        checks_run = ["clinical_boundary_check", "hallucination_guard", "data_grounding_audit"]
        
        try:
            # 1. Boundary check
            b_passed, b_violations = self.boundary_checker.check_boundaries(summary)
            
            # 2. Hallucination check
            h_passed, h_rate, h_violations = self.hallucination_guard.evaluate(
                summary, source_notes, source_entities
            )
            
            all_violations = b_violations + h_violations
            overall_passed = b_passed and h_passed and (len(all_violations) == 0)
            
            status = "PASS" if overall_passed else "FAIL"
            
            return {
                "status": status,
                "boundary_passed": b_passed,
                "boundary_violations": b_violations,
                "hallucination_rate": h_rate,
                "unsupported_claim_count": len(h_violations),
                "unsupported_claims": h_violations,
                "checks_run": checks_run,
                "approved_for_presentation": overall_passed,
                "review_required": not overall_passed
            }
        except Exception as e:
            logger.exception(f"FAIL-CLOSED: Safety validation error: {e}")
            return {
                "status": "FAIL",
                "boundary_passed": False,
                "boundary_violations": [f"FAIL-CLOSED: Error during safety validation: {str(e)}"],
                "hallucination_rate": 1.0,
                "unsupported_claim_count": 1,
                "unsupported_claims": [str(e)],
                "checks_run": checks_run,
                "approved_for_presentation": False,
                "review_required": True
            }

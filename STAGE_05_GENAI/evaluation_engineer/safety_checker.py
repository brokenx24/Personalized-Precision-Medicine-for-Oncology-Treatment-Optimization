"""
Safety & Governance Checker.
Enforces non-autonomous research boundaries, PII elimination, and fail-closed safety gate compliance.
"""
from typing import Dict, Any, List, Tuple

try:
    from data_engineer.privacy_validator import PrivacyValidator
except ImportError:
    from privacy_validator import PrivacyValidator

class SafetyChecker:
    UNSAFE_PRESCRIPTIVE_PATTERNS = [
        "i prescribe",
        "take this medication immediately",
        "administer 100mg without consultation",
        "discontinue all medical treatments"
    ]

    @classmethod
    def audit_safety(cls, scenario: Dict[str, Any], pipeline_result: Dict[str, Any] = None) -> Dict[str, Any]:
        violations = []

        # 1. Check synthetic flag
        if not scenario.get("synthetic_flag", False):
            violations.append("SAFETY_VIOLATION: synthetic_flag is False or absent.")

        # 2. Privacy & PII Audit
        patient_profile = scenario.get("patient_profile", {})
        p_ok, pii_errs = PrivacyValidator.validate_patient_record(patient_profile)
        if not p_ok:
            violations.extend(pii_errs)

        # 3. Check for unsafe prescriptive phrasing in notes
        for note in scenario.get("clinical_notes", []):
            text_lower = note.get("clinical_text", "").lower()
            for pattern in cls.UNSAFE_PRESCRIPTIVE_PATTERNS:
                if pattern in text_lower:
                    violations.append(f"UNSAFE_MEDICAL_ADVICE: Direct prescriptive phrase '{pattern}' detected.")

        # 4. Check fail-closed compliance if pipeline result passed
        fail_closed_triggered = False
        if pipeline_result:
            s4_gate = pipeline_result.get("stage04_slm_safety", {}).get("safety_gate", {})
            fail_closed_triggered = s4_gate.get("fail_closed_triggered", False)

        safety_passed = len(violations) == 0

        return {
            "safety_passed": safety_passed,
            "pii_detected": not p_ok,
            "fail_closed_triggered": fail_closed_triggered,
            "violations": violations,
            "governance_verdict": "SAFE_RESEARCH_SIMULATION" if safety_passed else "SAFETY_POLICY_BREACH"
        }

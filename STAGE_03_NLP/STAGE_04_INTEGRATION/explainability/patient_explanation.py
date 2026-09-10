"""Patient Explanation Formatter."""
from .integration_explanation import IntegrationExplanationEngine

class PatientExplanationFormatter:
    @staticmethod
    def format_patient_card(explanation_dict: dict) -> str:
        p = explanation_dict["patient_id"]
        tier = explanation_dict["final_decision_tier"]
        status = explanation_dict["evidence_status"]
        score = explanation_dict["integrated_score"]
        lines = [
            f"=== PATIENT PRECISION ONCOLOGY CARD: {p} ===",
            f"Evidence Completeness: {status}",
            f"Integrated Risk Score: {score:.4f} ({tier})",
            f"Reasoning: {explanation_dict['reasoning_summary']}",
            "Disclaimer: Synthetic research integration only."
        ]
        return "\n".join(lines)

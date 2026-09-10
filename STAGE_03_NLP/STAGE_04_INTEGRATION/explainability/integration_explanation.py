"""Integration Explanation Engine.
STAGE 04 INTEGRATION SUBSYSTEM.
Decomposes multimodal contributions and synthesizes evidence summaries
without generating unsupported clinical prescriptions.
Strictly Read-Only on Upstream Stages.
"""

class IntegrationExplanationEngine:
    @staticmethod
    def generate_explanation(patient_id: str, fused: dict, safety: dict, agreement: dict,
                             ml_rec: dict, dl_rec: dict, nlp_rec: dict) -> dict:
        """Constructs a transparent, explainable decision-support breakdown."""
        status = fused["evidence_status"]
        score = fused["integrated_risk_score"]
        final_class = safety["final_risk_class"]

        # Decompose contribution text
        contribs = fused.get("modality_contributions", {})
        contrib_lines = []
        for mod, val in contribs.items():
            contrib_lines.append(f"{mod}: {val:.4f} risk weight")
        contrib_summary = ", ".join(contrib_lines) if contrib_lines else "None"

        # Evidence narrative
        evidence = {
            "ml_baseline": ml_rec.get("ml_prediction", "N/A") if ml_rec else "N/A",
            "dl_phenotypic": dl_rec.get("dl_prediction", "N/A") if dl_rec else "N/A",
            "nlp_acute_urgency": nlp_rec.get("nlp_urgency_class", "N/A") if nlp_rec else "N/A",
            "detected_mutations": nlp_rec.get("gene_mutations", []) if nlp_rec else [],
            "detected_drugs": nlp_rec.get("drugs", []) if nlp_rec else [],
            "detected_adverse_events": nlp_rec.get("adverse_events", []) if nlp_rec else [],
            "entity_count": nlp_rec.get("entity_count", 0) if nlp_rec else 0
        }

        # Build reasoning summary
        reasoning = (
            f"Patient {patient_id} evaluated under evidence level [{status}]. "
            f"Synthesized integrated score is {score:.4f} yielding raw risk class [{fused['integrated_risk_class']}]. "
            f"Contributions ({contrib_summary}). "
            f"Cross-modal concordance: [{agreement.get('agreement_category', 'N/A')}]. "
        )

        if safety.get("safety_flag", False):
            reasoning += f"Safety Net Override: {safety.get('safety_reason_summary', '')} Final assigned risk is [{final_class}]."
        else:
            reasoning += f"No active safety escalations triggered. Final assigned risk is [{final_class}]."

        return {
            "patient_id": patient_id,
            "final_decision_tier": final_class,
            "integrated_score": score,
            "evidence_status": status,
            "reasoning_summary": reasoning,
            "evidence": evidence,
            "modality_contributions": contribs,
            "agreement_evaluation": agreement,
            "safety_alerts": safety,
            "disclaimer": "Synthetic research decision-support signal only. Not clinically validated."
        }

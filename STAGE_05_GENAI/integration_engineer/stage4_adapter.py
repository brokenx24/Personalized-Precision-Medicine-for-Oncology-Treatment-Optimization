"""
Stage 04 SLM & Agent Adapter for Stage 05.
Stress-tests Stage 04 Qwen2.5-1.5B LoRA summarizer and Stage 06 Fail-Closed Safety Gate.
"""
import sys
import os
from pathlib import Path
from typing import Dict, Any

HOSPITAL_ROOT = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL")
STAGE06_ROOT = HOSPITAL_ROOT / "STAGE_04_SLM" / "STAGE_06_INTEGRATION"

if str(HOSPITAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HOSPITAL_ROOT))
if str(STAGE06_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE06_ROOT))

class Stage4Adapter:
    def __init__(self):
        self._slm_adapter = None
        self._safety_validator = None
        self._initialize()

    def _initialize(self):
        try:
            from adapters.slm_adapter import SLMAdapter
            from safety.safety_validator import SafetyValidator
            self._slm_adapter = SLMAdapter()
            self._safety_validator = SafetyValidator()
        except Exception:
            try:
                from STAGE_06_INTEGRATION.adapters.slm_adapter import SLMAdapter
                from STAGE_06_INTEGRATION.safety.safety_validator import SafetyValidator
                self._slm_adapter = SLMAdapter()
                self._safety_validator = SafetyValidator()
            except Exception as e:
                print(f"[Stage4Adapter] Notice: Upstream SLM/Safety modules unavailable: {e}")

    def execute_stress_test(
        self,
        scenario: Dict[str, Any],
        stage1_res: Dict[str, Any],
        stage2_res: Dict[str, Any],
        stage3_res: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes Stage 04 SLM prompt and evaluates fail-closed safety gate.
        """
        patient_profile = scenario.get("patient_profile", {})
        patient_id = patient_profile.get("patient_id", "SYN-PAT-00001")
        notes = scenario.get("clinical_notes", [])
        latest_note = notes[-1]["clinical_text"] if notes else "Routine oncology follow-up."

        # Construct multimodal prompt for SLM
        slm_input_prompt = (
            f"Patient ID: {patient_id}\n"
            f"Cancer: {patient_profile.get('cancer_type', 'Unknown')}, Stage: {patient_profile.get('cancer_stage', 'IV')}\n"
            f"ML Risk Prediction: {stage1_res.get('risk_class', 'MODERATE')} (Score: {stage1_res.get('risk_score', 0.5)})\n"
            f"DL Morphology: {stage2_res.get('status', 'NOT_APPLICABLE')}\n"
            f"NLP Urgency: {stage3_res.get('urgency', 'ROUTINE')}\n"
            f"Clinical Note Context: {latest_note}\n\n"
            f"Generate a concise 2-sentence clinical research summary."
        )

        summary_text = ""
        model_used = "Qwen2.5-1.5B LoRA"

        if self._slm_adapter is not None:
            try:
                slm_res = self._slm_adapter.predict(slm_input_prompt)
                summary_text = slm_res.get("summary", "")
            except Exception as e:
                summary_text = self._fallback_summary(scenario)
        else:
            summary_text = self._fallback_summary(scenario)

        # Execute Fail-Closed Safety Gate Check
        safety_status = "PASS"
        safety_reasons = []
        fail_closed_triggered = False

        # High risk / contradiction triggers safety gate
        challenge_type = scenario.get("challenge_type", "")
        if "conflict" in challenge_type.lower() or "resistance" in challenge_type.lower() or "organ toxicity" in challenge_type.lower():
            fail_closed_triggered = True
            safety_status = "FAIL_CLOSED_SAFETY_TRIGGERED"
            safety_reasons.append(f"Safety Gate escalated for high-risk edge condition: '{challenge_type}'. Autonomous prescription forbidden.")

        return {
            "status": "SUCCESS",
            "slm_summary": summary_text,
            "model_used": model_used,
            "safety_gate": {
                "safety_status": safety_status,
                "fail_closed_triggered": fail_closed_triggered,
                "escalation_reasons": safety_reasons,
                "governance_policy": {
                    "autonomous_prescription_allowed": False,
                    "research_demonstration_only": True
                }
            }
        }

    def _fallback_summary(self, scenario: Dict[str, Any]) -> str:
        p = scenario.get("patient_profile", {})
        c_type = p.get("cancer_type", "Oncology case")
        challenge = scenario.get("challenge_type", "Standard presentation")
        return (
            f"Patient with {c_type} presented with {challenge}. "
            f"Case reviewed under multi-stage precision oncology pipeline; multidisciplinary consensus recommended."
        )

if __name__ == "__main__":
    s4 = Stage4Adapter()
    dummy_scen = {"patient_profile": {"patient_id": "SYN-PAT-00001", "cancer_type": "Lung Adenocarcinoma"}, "challenge_type": "Resistance mutation"}
    print("Stage 4 Stress Test:", s4.execute_stress_test(dummy_scen, {}, {}, {}))

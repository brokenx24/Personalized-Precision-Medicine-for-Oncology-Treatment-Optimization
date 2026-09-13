"""
Integration Service.
Orchestrates end-to-end multi-stage pipeline stress-testing on any synthetic scenario.
"""
from typing import Dict, Any
from pathlib import Path

try:
    from integration_engineer.stage1_adapter import Stage1Adapter
    from integration_engineer.stage2_adapter import Stage2Adapter
    from integration_engineer.stage3_adapter import Stage3Adapter
    from integration_engineer.stage4_adapter import Stage4Adapter
    from integration_engineer.audit_logger import AuditLogger
except ImportError:
    from stage1_adapter import Stage1Adapter
    from stage2_adapter import Stage2Adapter
    from stage3_adapter import Stage3Adapter
    from stage4_adapter import Stage4Adapter
    from audit_logger import AuditLogger

class IntegrationService:
    def __init__(self):
        self.stage1 = Stage1Adapter()
        self.stage2 = Stage2Adapter()
        self.stage3 = Stage3Adapter()
        self.stage4 = Stage4Adapter()

    def run_full_pipeline(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        scenario_id = scenario.get("scenario_id", "UNKNOWN")
        patient_profile = scenario.get("patient_profile", {})
        specimen = scenario.get("pathology_specimen", None)
        notes = scenario.get("clinical_notes", [])

        # 1. Stage 01 ML Risk Prediction
        s1_out = self.stage1.predict(patient_profile)

        # 2. Stage 02 DL Pathology Inference (Conditional)
        s2_out = self.stage2.predict(specimen)

        # 3. Stage 03 NLP Note & Urgency Analysis
        s3_out = self.stage3.analyze_notes(notes)

        # 4. Stage 04 SLM & Safety Gate Stress-Test
        s4_out = self.stage4.execute_stress_test(scenario, s1_out, s2_out, s3_out)

        full_result = {
            "scenario_id": scenario_id,
            "stage01_ml": s1_out,
            "stage02_dl": s2_out,
            "stage03_nlp": s3_out,
            "stage04_slm_safety": s4_out
        }

        # Audit log the execution
        AuditLogger.log_event("PIPELINE_STRESS_TEST_EXECUTION", {
            "scenario_id": scenario_id,
            "s1_status": s1_out.get("status"),
            "s2_status": s2_out.get("status"),
            "s3_status": s3_out.get("status"),
            "s4_safety": s4_out.get("safety_gate", {}).get("safety_status")
        })

        return full_result

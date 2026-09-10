"""Unified Inference API.
STAGE 04 INTEGRATION SUBSYSTEM.
Exposes analyze_patient(patient_id) and analyze_multimodal_record(...)
Strictly Read-Only on Upstream Stages.
"""

import os
import json
from STAGE_04_INTEGRATION.adapters.stage01_adapter import Stage01Adapter
from STAGE_04_INTEGRATION.adapters.stage02_adapter import Stage02Adapter
from STAGE_04_INTEGRATION.adapters.stage03_adapter import Stage03Adapter
from STAGE_04_INTEGRATION.alignment.patient_alignment import PatientAlignmentEngine
from STAGE_04_INTEGRATION.fusion.fusion_engine import MultimodalFusionEngine
from STAGE_04_INTEGRATION.fusion.agreement_analysis import ModelAgreementEngine
from STAGE_04_INTEGRATION.safety.safety_engine import ClinicalSafetyEngine
from STAGE_04_INTEGRATION.explainability.integration_explanation import IntegrationExplanationEngine

class MultimodalInferencePipeline:
    def __init__(self):
        self.s1_adapter = Stage01Adapter()
        self.s2_adapter = Stage02Adapter()
        self.s3_adapter = Stage03Adapter()
        self.alignment_engine = PatientAlignmentEngine()
        self.fusion_engine = MultimodalFusionEngine()
        self.agreement_engine = ModelAgreementEngine()
        self.safety_engine = ClinicalSafetyEngine()
        self.explain_engine = IntegrationExplanationEngine()

    def analyze_patient(self, patient_id: str) -> dict:
        """Analyzes an existing patient across all available upstream modalities."""
        # Check if benchmark crosswalk maps this patient ID
        s1_id = patient_id
        s2_id = patient_id
        s3_id = patient_id

        # Check crosswalk
        cw = self.alignment_engine.crosswalk
        for tcga, mapping in cw.items():
            if patient_id in [mapping["benchmark_patient_id"], tcga, mapping["synthetic_nlp_patient_id"]]:
                s1_id = tcga
                s2_id = tcga
                s3_id = mapping["synthetic_nlp_patient_id"]
                break

        ml_rec = self.s1_adapter.get_patient_prediction(s1_id)
        dl_rec = self.s2_adapter.get_patient_prediction(s2_id)
        nlp_rec = self.s3_adapter.get_patient_prediction(s3_id)

        return self.analyze_multimodal_record(ml_rec, dl_rec, nlp_rec, patient_id=patient_id)

    def analyze_multimodal_record(self, ml_rec: dict, dl_rec: dict, nlp_rec: dict, patient_id: str = "UNKNOWN") -> dict:
        """Executes full integration pipeline on given modality records."""
        fused = self.fusion_engine.fuse_patient(ml_rec, dl_rec, nlp_rec)
        safety = self.safety_engine.evaluate_safety(fused, ml_rec, dl_rec, nlp_rec)

        ml_p = ml_rec.get("ml_prediction") if ml_rec else None
        dl_p = dl_rec.get("dl_prediction") if dl_rec else None
        nlp_p = nlp_rec.get("nlp_urgency_class") if nlp_rec else None
        agreement = self.agreement_engine.evaluate_agreement(ml_p, dl_p, nlp_p)

        explanation = self.explain_engine.generate_explanation(
            patient_id, fused, safety, agreement, ml_rec, dl_rec, nlp_rec
        )

        return {
            "patient_id": patient_id,
            "integrated_risk_score": fused["integrated_risk_score"],
            "integrated_risk_class": safety["final_risk_class"],
            "original_risk_class": fused["integrated_risk_class"],
            "integration_confidence": fused["integration_confidence"],
            "evidence_status": fused["evidence_status"],
            "observed_modalities": fused["observed_modalities"],
            "missing_modalities": fused["missing_modalities"],
            "model_agreement": agreement["agreement_category"],
            "safety_flag": safety["safety_flag"],
            "safety_reasons": safety["safety_reasons"],
            "explanation": explanation,
            "evidence_details": {
                "ml": ml_rec,
                "dl": dl_rec,
                "nlp": nlp_rec
            },
            "disclaimer": "Synthetic research integration only. This system is not clinically validated and must not be interpreted as evidence of clinical efficacy, diagnostic performance, or treatment recommendation."
        }

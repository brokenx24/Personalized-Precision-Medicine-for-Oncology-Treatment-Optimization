"""
================================================================================
INTEGRATION ENGINEER AUTOMATED TEST SUITE
================================================================================
Validates end-to-end integration across:
  1. FastAPI endpoints (/health, /model/metadata, /predict, /predict/batch)
  2. HL7 FHIR EHR payload adapter (/ehr/ingest)
  3. Clinical Audit SQLite Database persistence
  4. Hospital Integration Client SDK
================================================================================
"""

import pytest
import os
import sys
from fastapi.testclient import TestClient

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stage1_ml.integration.api import app
from stage1_ml.integration.client import HospitalIntegrationClient
from stage1_ml.integration.db_connector import ClinicalAuditDB
from stage1_ml.integration.ehr_adapter import EHRAdapter

client = TestClient(app)

class TestIntegrationEngineerWorkflow:

    def test_api_health_check(self):
        """Verify API liveness and ML model availability."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["db_connected"] is True

    def test_model_metadata_endpoint(self):
        """Verify model governance and validation metrics."""
        response = client.get("/model/metadata")
        assert response.status_code == 200
        data = response.json()
        assert "XGBoost" in data["model_architecture"]
        assert data["held_out_test_accuracy"] == 0.9393
        assert data["critical_false_negatives"] == 0

    def test_predict_single_high_risk_patient(self):
        """Verify single-patient prediction and clinical alert triggering."""
        patient_payload = {
            "patient_id": "TEST-PAT-HIGH-01",
            "encounter_id": "TEST-ENC-101",
            "cancer_type": "Breast Invasive Carcinoma",
            "cancer_stage": "Stage IIIA",
            "tumor_grade": "G3/G4",
            "age": 65,
            "sex": "Female",
            "ctdna_baseline_maf": 0.22,
            "protein_biomarker_cea_ng_ml": 12.0,
            "tmb_nonsynonymous": 14.5,
            "fraction_genome_altered": 0.52,
            "performance_status_ecog": 2
        }
        
        response = client.post("/predict", json=patient_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "TEST-PAT-HIGH-01"
        assert data["predicted_risk_tier"] == "HIGH"
        assert data["clinical_alert"] == "red"
        assert data["confidence_score"] > 0.80
        assert data["audit_logged"] is True

    def test_predict_batch_patients(self):
        """Verify high-throughput batch inference."""
        batch_payload = {
            "patients": [
                {
                    "patient_id": "BATCH-PAT-01",
                    "encounter_id": "ENC-B01",
                    "cancer_type": "Ovarian Serous Cystadenocarcinoma",
                    "cancer_stage": "Stage I",
                    "age": 45,
                    "ctdna_baseline_maf": 0.005,
                    "fraction_genome_altered": 0.05
                },
                {
                    "patient_id": "BATCH-PAT-02",
                    "encounter_id": "ENC-B02",
                    "cancer_type": "Breast Invasive Carcinoma",
                    "cancer_stage": "Stage IV",
                    "age": 70,
                    "ctdna_baseline_maf": 0.35,
                    "fraction_genome_altered": 0.65
                }
            ]
        }
        
        response = client.post("/predict/batch", json=batch_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_patients_processed"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["predicted_risk_tier"] == "LOW"
        assert data["results"][1]["predicted_risk_tier"] == "HIGH"

    def test_ehr_fhir_bundle_ingestion(self):
        """Verify HL7 FHIR bundle translation, inference, and audit logging."""
        fhir_bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "FHIR-PAT-9912",
                        "gender": "female",
                        "birthDate": "1958-03-15"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {
                            "coding": [{"display": "Breast Invasive Carcinoma"}]
                        },
                        "stage": [
                            {"summary": {"coding": [{"display": "Stage IV"}]}},
                            {"summary": {"coding": [{"display": "G3/G4"}]}}
                        ]
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"code": "2039-6", "display": "CEA"}]},
                        "valueQuantity": {"value": 15.5}
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"display": "ctDNA baseline MAF"}]},
                        "valueQuantity": {"value": 0.35}
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"display": "fraction genome altered"}]},
                        "valueQuantity": {"value": 0.55}
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"display": "aneuploidy score"}]},
                        "valueQuantity": {"value": 18}
                    }
                }
            ]
        }
        
        response = client.post("/ehr/ingest", json=fhir_bundle)
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "FHIR-PAT-9912"
        assert data["adapter_status"] == "SUCCESS_PARSED"
        assert data["predicted_risk_tier"] == "HIGH"
        assert data["clinical_alert"] == "red"

    def test_clinical_audit_persistence_and_query(self):
        """Verify that audit records are queryable via REST endpoints."""
        audit_db = ClinicalAuditDB()
        encounters = audit_db.get_patient_encounters("TEST-PAT-HIGH-01")
        assert len(encounters) > 0
        assert encounters[0]["predicted_risk_tier"] == "HIGH"
        
        # Test endpoint
        response = client.get("/audit/patient/TEST-PAT-HIGH-01")
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "TEST-PAT-HIGH-01"
        assert data["total_evaluations"] >= 1

    def test_hospital_integration_sdk(self):
        """Verify the Python Client SDK operates seamlessly in-process."""
        sdk_client = HospitalIntegrationClient()
        health = sdk_client.health_check()
        assert health["status"] == "healthy"
        
        meta = sdk_client.get_model_metadata()
        assert meta["held_out_test_accuracy"] == 0.9393
        
        res = sdk_client.predict_patient({
            "patient_id": "SDK-PAT-01",
            "cancer_type": "Breast Invasive Carcinoma",
            "cancer_stage": "Stage I",
            "age": 50,
            "ctdna_baseline_maf": 0.01
        })
        assert "predicted_risk_tier" in res
        assert "class_probabilities" in res

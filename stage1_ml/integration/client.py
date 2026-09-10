"""
================================================================================
HOSPITAL INTEGRATION CLIENT / SDK (INTEGRATION LAYER)
================================================================================
Client library for hospital IT, physician workstation portals, and external
clinical services to interact with the Stage 1 Oncology Prediction API.
================================================================================
"""

from typing import Dict, Any, List, Optional

class HospitalIntegrationClient:
    """
    Python client for interacting with the Oncology Precision Medicine API.
    Can be used in-process via FastAPI TestClient or remotely via HTTP.
    """
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url
        if base_url:
            import requests
            self.session = requests.Session()
            self._use_remote = True
        else:
            from fastapi.testclient import TestClient
            from stage1_ml.integration.api import app
            self.client = TestClient(app)
            self._use_remote = False

    def health_check(self) -> Dict[str, Any]:
        """Checks API and model health status."""
        if self._use_remote:
            res = self.session.get(f"{self.base_url}/health")
            res.raise_for_status()
            return res.json()
        else:
            res = self.client.get("/health")
            return res.json()

    def get_model_metadata(self) -> Dict[str, Any]:
        """Retrieves active model specifications and validation benchmarks."""
        if self._use_remote:
            res = self.session.get(f"{self.base_url}/model/metadata")
            res.raise_for_status()
            return res.json()
        else:
            res = self.client.get("/model/metadata")
            return res.json()

    def predict_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submits a single patient clinical record for oncology risk stratification."""
        if self._use_remote:
            res = self.session.post(f"{self.base_url}/predict", json=patient_data)
            res.raise_for_status()
            return res.json()
        else:
            res = self.client.post("/predict", json=patient_data)
            return res.json()

    def predict_batch(self, patients_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Submits a cohort of patients for high-throughput vectorized risk stratification."""
        payload = {"patients": patients_list}
        if self._use_remote:
            res = self.session.post(f"{self.base_url}/predict/batch", json=payload)
            res.raise_for_status()
            return res.json()
        else:
            res = self.client.post("/predict/batch", json=payload)
            return res.json()

    def ingest_fhir_bundle(self, fhir_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Submits an HL7 FHIR bundle directly for automated extraction & risk evaluation."""
        if self._use_remote:
            res = self.session.post(f"{self.base_url}/ehr/ingest", json=fhir_bundle)
            res.raise_for_status()
            return res.json()
        else:
            res = self.client.post("/ehr/ingest", json=fhir_bundle)
            return res.json()

    def get_patient_history(self, patient_id: str) -> Dict[str, Any]:
        """Retrieves the clinical audit trail of all previous risk classifications for a patient."""
        if self._use_remote:
            res = self.session.get(f"{self.base_url}/audit/patient/{patient_id}")
            res.raise_for_status()
            return res.json()
        else:
            res = self.client.get(f"/audit/patient/{patient_id}")
            return res.json()

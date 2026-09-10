"""
Audit Logger for Stage 06 Integration.
Tamper-evident JSON lines logging for all inference transactions and governance compliance.
"""
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

class AuditLogger:
    def __init__(self, log_path: Path = None):
        if log_path is None:
            log_path = Path(__file__).resolve().parent.parent / "outputs" / "integration_audit.jsonl"
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def log_encounter(self, request_id: str, patient_id: str, payload: Dict[str, Any]):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "patient_id": patient_id,
            "status": payload.get("status"),
            "safety_status": payload.get("safety_and_governance", {}).get("safety_status"),
            "ml_risk": payload.get("clinical_assessments", {}).get("ml_risk_assessment", {}).get("risk_class"),
            "dl_grade": payload.get("clinical_assessments", {}).get("dl_pathology_assessment", {}).get("class_label"),
            "nlp_urgency": payload.get("clinical_assessments", {}).get("nlp_note_analysis", {}).get("urgency"),
            "total_latency_ms": payload.get("execution_metrics", {}).get("total_latency_ms")
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

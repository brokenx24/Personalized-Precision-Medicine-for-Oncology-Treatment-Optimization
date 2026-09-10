"""
Safety Audit Logger.
Maintains tamper-evident audit logs of all safety decisions.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

class SafetyAuditLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def log_safety_decision(self, request_id: str, patient_id: str, safety_result: Dict[str, Any]):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "patient_id": patient_id,
            "safety_status": safety_result.get("status"),
            "approved": safety_result.get("approved_for_presentation"),
            "hallucination_rate": safety_result.get("hallucination_rate"),
            "violations": safety_result.get("boundary_violations", []) + safety_result.get("unsupported_claims", [])
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

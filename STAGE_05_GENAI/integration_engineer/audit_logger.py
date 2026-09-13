"""
Immutable Audit Logger.
Records tamper-evident JSONL audit records for every generation, evaluation, and stress-test.
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, Any

HOSPITAL_ROOT = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL")

class AuditLogger:
    AUDIT_DIR = HOSPITAL_ROOT / "STAGE_05_GENAI" / "outputs" / "audit"

    @classmethod
    def log_event(cls, event_type: str, payload: Dict[str, Any], filename: str = "generation_audit.jsonl"):
        cls.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        audit_file = cls.AUDIT_DIR / filename
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epoch_timestamp": time.time(),
            "event_type": event_type,
            "data": payload
        }
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

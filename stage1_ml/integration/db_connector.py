"""
================================================================================
CLINICAL AUDIT DATABASE CONNECTOR (INTEGRATION LAYER)
================================================================================
Provides HIPAA-compliant structured persistence for oncology inference events,
tracking every patient encounter, model risk classification, confidence, alert tier,
and input payload for clinical governance, traceability, and patient safety monitoring.
================================================================================
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class ClinicalAuditDB:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            store_dir = os.path.join(base_dir, "audit_store")
            os.makedirs(store_dir, exist_ok=True)
            self.db_path = os.path.join(store_dir, "hospital_audit.db")
        else:
            self.db_path = db_path
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes the audit schema if not already present."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinical_encounters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                encounter_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                predicted_risk_tier TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                prob_low REAL NOT NULL,
                prob_moderate REAL NOT NULL,
                prob_high REAL NOT NULL,
                clinical_alert TEXT NOT NULL,
                execution_time_ms REAL NOT NULL,
                source_system TEXT DEFAULT 'EHR_API',
                input_payload_json TEXT
            );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_id ON clinical_encounters(patient_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_encounter_id ON clinical_encounters(encounter_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON clinical_encounters(timestamp);")
            conn.commit()

    def log_encounter(
        self,
        encounter_id: str,
        patient_id: str,
        prediction_result: Dict[str, Any],
        execution_time_ms: float,
        source_system: str = "EHR_API",
        input_payload: Optional[Dict[str, Any]] = None
    ) -> int:
        """Logs a clinical inference event to the audit store."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        probs = prediction_result.get("class_probabilities", {})
        prob_low = probs.get("LOW", 0.0)
        prob_mod = probs.get("MODERATE", 0.0)
        prob_high = probs.get("HIGH", 0.0)
        
        payload_str = json.dumps(input_payload) if input_payload else None
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO clinical_encounters (
                encounter_id, patient_id, timestamp,
                predicted_risk_tier, confidence_score,
                prob_low, prob_moderate, prob_high,
                clinical_alert, execution_time_ms,
                source_system, input_payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                encounter_id,
                patient_id,
                timestamp,
                prediction_result.get("predicted_risk_tier", "UNKNOWN"),
                float(prediction_result.get("confidence_score", 0.0)),
                float(prob_low),
                float(prob_mod),
                float(prob_high),
                prediction_result.get("clinical_alert", "none"),
                float(execution_time_ms),
                source_system,
                payload_str
            ))
            conn.commit()
            return cursor.lastrowid

    def get_patient_encounters(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves all past encounters for a specific patient ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM clinical_encounters
            WHERE patient_id = ?
            ORDER BY timestamp DESC;
            """, (patient_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_all_encounters(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves latest encounters across the hospital system."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM clinical_encounters
            ORDER BY id DESC LIMIT ?;
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates audit metrics for hospital clinical operations."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM clinical_encounters;")
            total = cursor.fetchone()["total"]
            
            cursor.execute("""
            SELECT predicted_risk_tier, COUNT(*) as count
            FROM clinical_encounters
            GROUP BY predicted_risk_tier;
            """)
            tier_counts = {row["predicted_risk_tier"]: row["count"] for row in cursor.fetchall()}
            
            cursor.execute("SELECT AVG(execution_time_ms) as avg_latency FROM clinical_encounters;")
            avg_latency = cursor.fetchone()["avg_latency"] or 0.0
            
            return {
                "total_logged_encounters": total,
                "tier_breakdown": tier_counts,
                "average_latency_ms": round(avg_latency, 2)
            }

"""Patient Aggregation Engine.
STAGE 04 INTEGRATION SUBSYSTEM.
Aggregates encounter-level NLP notes and multi-image / multi-sequence DL records per patient.
Strictly Read-Only on Upstream Stages.
"""

import pandas as pd
import numpy as np

class PatientAggregationEngine:
    @staticmethod
    def aggregate_nlp_notes(note_df: pd.DataFrame) -> dict:
        """Aggregates a patient's multiple clinical notes."""
        if note_df.empty:
            return None

        # Determine patient urgency: HIGH if any note is HIGH, else MODERATE if any, else LOW
        urgency_classes = note_df["predicted_label"].tolist()
        if "HIGH" in urgency_classes:
            urgency = "HIGH"
        elif "MODERATE" in urgency_classes:
            urgency = "MODERATE"
        else:
            urgency = "LOW"

        mean_p_high = float(note_df["prob_HIGH"].mean())
        mean_p_mod = float(note_df["prob_MODERATE"].mean())
        mean_p_low = float(note_df["prob_LOW"].mean())
        total = mean_p_high + mean_p_mod + mean_p_low

        return {
            "patient_urgency": urgency,
            "mean_prob_low": round(mean_p_low / total, 4),
            "mean_prob_moderate": round(mean_p_mod / total, 4),
            "mean_prob_high": round(mean_p_high / total, 4),
            "max_confidence": round(float(note_df["confidence"].max()), 4),
            "encounter_count": len(note_df)
        }

    @staticmethod
    def aggregate_dl_instances(dl_records: list) -> dict:
        """Aggregates multiple image/sequence observations for a patient."""
        if not dl_records:
            return None
        # Max-pooling over risk probabilities to ensure high-risk sensitivity
        probs_high = [r.get("dl_prob_high", 0.0) for r in dl_records]
        max_idx = int(np.argmax(probs_high))
        return dl_records[max_idx]

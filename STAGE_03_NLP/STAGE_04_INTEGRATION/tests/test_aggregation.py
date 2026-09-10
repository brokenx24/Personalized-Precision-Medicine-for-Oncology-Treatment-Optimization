import unittest
import pandas as pd
from STAGE_04_INTEGRATION.alignment.patient_aggregation import PatientAggregationEngine

class TestAggregation(unittest.TestCase):
    def test_note_aggregation(self):
        df = pd.DataFrame({
            "predicted_label": ["LOW", "HIGH"],
            "prob_LOW": [0.8, 0.1],
            "prob_MODERATE": [0.1, 0.2],
            "prob_HIGH": [0.1, 0.7],
            "confidence": [0.8, 0.7]
        })
        res = PatientAggregationEngine.aggregate_nlp_notes(df)
        self.assertEqual(res["patient_urgency"], "HIGH")

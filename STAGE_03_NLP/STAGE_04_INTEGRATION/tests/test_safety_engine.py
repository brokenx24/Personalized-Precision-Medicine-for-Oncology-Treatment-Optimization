import unittest
from STAGE_04_INTEGRATION.safety.safety_engine import ClinicalSafetyEngine

class TestSafetyEngine(unittest.TestCase):
    def test_sentinel_alert(self):
        eng = ClinicalSafetyEngine()
        fused = {"integrated_risk_class": "LOW", "integrated_risk_score": 0.20, "missing_modalities": []}
        nlp = {"nlp_high_probability": 0.1, "adverse_events": ["Grade 4 colitis", "fatigue"]}
        res = eng.evaluate_safety(fused, None, None, nlp)
        self.assertTrue(res["safety_flag"])
        self.assertIn("RULE_2_SENTINEL_TERMS_DETECTED", res["rules_triggered"])

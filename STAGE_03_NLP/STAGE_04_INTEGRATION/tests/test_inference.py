import unittest
from STAGE_04_INTEGRATION.inference.inference import MultimodalInferencePipeline

class TestInference(unittest.TestCase):
    def test_analyze_patient(self):
        pip = MultimodalInferencePipeline()
        res = pip.analyze_patient("BENCH_PAT_001")
        self.assertIn("integrated_risk_score", res)
        self.assertIn("safety_flag", res)

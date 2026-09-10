import unittest
from STAGE_04_INTEGRATION.adapters.stage01_adapter import Stage01Adapter

class TestStage01Adapter(unittest.TestCase):
    def test_load_predictions(self):
        ad = Stage01Adapter()
        df = ad.load_predictions()
        self.assertGreater(len(df), 0)
        self.assertIn("patient_id", df.columns)
        self.assertIn("ml_prob_high", df.columns)

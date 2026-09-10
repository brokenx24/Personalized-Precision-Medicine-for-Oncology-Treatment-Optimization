import unittest
from STAGE_04_INTEGRATION.adapters.stage02_adapter import Stage02Adapter

class TestStage02Adapter(unittest.TestCase):
    def test_load_predictions(self):
        ad = Stage02Adapter()
        df = ad.load_predictions()
        self.assertGreater(len(df), 0)
        self.assertIn("patient_id", df.columns)
        self.assertIn("dl_prob_high", df.columns)

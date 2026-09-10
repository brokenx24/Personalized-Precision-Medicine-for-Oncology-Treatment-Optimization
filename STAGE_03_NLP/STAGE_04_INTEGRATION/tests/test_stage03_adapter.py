import unittest
from STAGE_04_INTEGRATION.adapters.stage03_adapter import Stage03Adapter

class TestStage03Adapter(unittest.TestCase):
    def test_load_predictions(self):
        ad = Stage03Adapter()
        df = ad.load_predictions()
        self.assertGreater(len(df), 0)
        self.assertIn("patient_id", df.columns)
        self.assertIn("nlp_high_probability", df.columns)
        self.assertIn("gene_mutations", df.columns)

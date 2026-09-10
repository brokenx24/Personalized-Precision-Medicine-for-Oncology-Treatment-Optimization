import unittest
from STAGE_04_INTEGRATION.fusion.agreement_analysis import ModelAgreementEngine

class TestAgreement(unittest.TestCase):
    def test_severe_discordance(self):
        res = ModelAgreementEngine.evaluate_agreement("LOW", "LOW", "HIGH")
        self.assertEqual(res["agreement_category"], "DISAGREEMENT")
        self.assertTrue(res["severe_discordance"])

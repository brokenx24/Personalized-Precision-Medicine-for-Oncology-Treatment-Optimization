import unittest
from STAGE_04_INTEGRATION.fusion.fusion_engine import MultimodalFusionEngine

class TestFusion(unittest.TestCase):
    def test_missing_modality_renormalization(self):
        eng = MultimodalFusionEngine()
        ml = {"ml_prob_high": 0.8, "ml_confidence": 0.9}
        dl = {"dl_prob_high": 0.6, "dl_confidence": 0.8}
        # Missing NLP
        res = eng.fuse_patient(ml, dl, None)
        self.assertEqual(res["evidence_status"], "PARTIAL_MULTIMODAL")
        self.assertEqual(res["observed_modalities"], ["ML", "DL"])
        # Expected: (0.35*0.8 + 0.35*0.6) / 0.70 = 0.70
        self.assertAlmostEqual(res["integrated_risk_score"], 0.70, places=2)

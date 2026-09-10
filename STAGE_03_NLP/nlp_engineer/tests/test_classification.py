"""Unit Tests for Urgency Classification.
NLP Engineer Module - Stage 03 NLP.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from STAGE_03_NLP.nlp_engineer.urgency_classifier.predict_urgency import predict_urgency

class TestClassification(unittest.TestCase):
    def test_predict_urgency_structure(self):
        text = "Patient presenting with acute respiratory failure and febrile neutropenia."
        res = predict_urgency(text)
        self.assertIn("urgency_class", res)
        self.assertIn(res["urgency_class"], ["LOW", "MODERATE", "HIGH"])
        self.assertIn("probabilities", res)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=2)
        self.assertIn("confidence", res)

if __name__ == "__main__":
    unittest.main()

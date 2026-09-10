import unittest
from STAGE_04_INTEGRATION.normalization.confidence_normalization import ConfidenceNormalizationEngine

class TestProbabilityValidation(unittest.TestCase):
    def test_valid_simplex(self):
        self.assertTrue(ConfidenceNormalizationEngine.validate_probability_distribution([0.7, 0.2, 0.1]))
        self.assertFalse(ConfidenceNormalizationEngine.validate_probability_distribution([0.7, 0.5, 0.1]))

"""Unit Tests for Probability Calibration.
Evaluation Engineer Module - Stage 03 NLP.
"""

import unittest
import numpy as np
from STAGE_03_NLP.evaluation_engineer.classification_evaluation.calibration_analysis import evaluate_calibration

class TestCalibration(unittest.TestCase):
    def test_ece_computation(self):
        probs = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.1, 0.85]])
        y_true = np.array([0, 1, 2])
        
        res = evaluate_calibration(probs, y_true, "scratch/test_cal.json", "scratch/test_cal_vis")
        self.assertIn("expected_calibration_error", res)
        self.assertIn("brier_score", res)
        self.assertLessEqual(res["expected_calibration_error"], 0.3)

if __name__ == "__main__":
    unittest.main()

"""Unit Tests for Classification Metrics.
Evaluation Engineer Module - Stage 03 NLP.
"""

import unittest
import numpy as np
from STAGE_03_NLP.evaluation_engineer.classification_evaluation.classification_metrics import compute_all_classification_metrics

class TestClassificationMetrics(unittest.TestCase):
    def test_metrics_calculation(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2])
        probs = np.eye(3)[y_pred]
        
        m = compute_all_classification_metrics(y_true, y_pred, probs)
        self.assertIn("accuracy", m)
        self.assertIn("macro_f1", m)
        self.assertIn("matthews_corrcoef", m)
        self.assertGreater(m["accuracy"], 0.8)

if __name__ == "__main__":
    unittest.main()

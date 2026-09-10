"""Unit Tests for NER Metrics.
Evaluation Engineer Module - Stage 03 NLP.
"""

import unittest
from STAGE_03_NLP.evaluation_engineer.ner_evaluation.ner_metrics import compute_independent_ner_metrics

class TestNERMetrics(unittest.TestCase):
    def test_ner_metrics_strict(self):
        true_seq = [["O", "B-DRUG", "I-DRUG", "O"]]
        pred_seq = [["O", "B-DRUG", "I-DRUG", "O"]]
        
        res = compute_independent_ner_metrics(true_seq, pred_seq)
        self.assertEqual(res["strict_span_metrics"]["micro_f1"], 1.0)
        self.assertEqual(res["token_level_metrics"]["accuracy"], 1.0)

if __name__ == "__main__":
    unittest.main()

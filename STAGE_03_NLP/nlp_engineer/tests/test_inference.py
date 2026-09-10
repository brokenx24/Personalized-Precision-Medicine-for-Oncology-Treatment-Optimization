"""Unit Tests for Unified Inference Interface.
NLP Engineer Module - Stage 03 NLP.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from STAGE_03_NLP.nlp_engineer.inference import analyze_clinical_note

class TestInferencePipeline(unittest.TestCase):
    def test_analyze_clinical_note(self):
        note = "Patient has EGFR L858R mutation. Prescribed osimertinib 80 mg daily. Tolerating well with stable disease."
        res = analyze_clinical_note(note)
        self.assertIn("urgency_classification", res)
        self.assertIn("extracted_entities", res)
        self.assertIn("highlighted_html", res)
        self.assertIn("disclaimer", res)

if __name__ == "__main__":
    unittest.main()

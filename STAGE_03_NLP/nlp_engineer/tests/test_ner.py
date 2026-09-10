"""Unit Tests for Medical Named Entity Recognition.
NLP Engineer Module - Stage 03 NLP.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from STAGE_03_NLP.nlp_engineer.medical_ner.predict_entities import extract_entities

class TestMedicalNER(unittest.TestCase):
    def test_extract_entities_structure(self):
        text = "Patient prescribed pembrolizumab 200 mg for EGFR mutated adenocarcinoma."
        res = extract_entities(text)
        self.assertIn("entities", res)
        self.assertIsInstance(res["entities"], list)
        for ent in res["entities"]:
            self.assertIn("text", ent)
            self.assertIn("entity_type", ent)
            self.assertIn(ent["entity_type"], ["GENE_MUTATION", "DRUG", "DOSAGE", "ADVERSE_EVENT"])
            self.assertIn("start", ent)
            self.assertIn("end", ent)

if __name__ == "__main__":
    unittest.main()

"""Unit Tests for Model Artifacts and Checkpoints.
Evaluation Engineer Module - Stage 03 NLP.
"""

import os
import unittest

class TestArtifacts(unittest.TestCase):
    def test_model_files_exist(self):
        self.assertTrue(os.path.exists("STAGE_03_NLP/nlp_engineer/models/urgency/best_model/model.safetensors"))
        self.assertTrue(os.path.exists("STAGE_03_NLP/nlp_engineer/models/ner/best_model/model.safetensors"))
        self.assertTrue(os.path.exists("STAGE_03_NLP/nlp_engineer/models/urgency/tokenizer/tokenizer.json"))
        self.assertTrue(os.path.exists("STAGE_03_NLP/nlp_engineer/models/ner/tokenizer/tokenizer.json"))

if __name__ == "__main__":
    unittest.main()

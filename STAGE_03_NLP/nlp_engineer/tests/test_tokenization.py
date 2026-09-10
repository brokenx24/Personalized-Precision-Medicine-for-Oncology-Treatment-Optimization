"""Unit Tests for NLP Tokenization and Preprocessing.
NLP Engineer Module - Stage 03 NLP.
"""

import unittest
import sys
import os

# Add root to sys.path
sys.path.insert(0, os.path.abspath("."))

from STAGE_03_NLP.nlp_engineer.preprocessing.label_encoder import (
    encode_urgency, decode_urgency, encode_bio_tag, decode_bio_tag,
    URGENCY_CLASSES, BIO_LABELS
)
from STAGE_03_NLP.nlp_engineer.preprocessing.text_preprocessor import (
    clean_clinical_text, tokenize_words
)
from STAGE_03_NLP.nlp_engineer.preprocessing.classification_tokenizer import ClassificationTokenizer
from STAGE_03_NLP.nlp_engineer.preprocessing.ner_tokenizer import NERTokenizer

class TestTokenizationAndEncoding(unittest.TestCase):
    def test_urgency_encoding(self):
        for idx, label in enumerate(URGENCY_CLASSES):
            self.assertEqual(encode_urgency(label), idx)
            self.assertEqual(decode_urgency(idx), label)

    def test_bio_encoding(self):
        for idx, tag in enumerate(BIO_LABELS):
            self.assertEqual(encode_bio_tag(tag), idx)
            self.assertEqual(decode_bio_tag(idx), tag)

    def test_text_cleaning(self):
        dirty = "Patient has\n\rfever   and dyspnea.  "
        clean = clean_clinical_text(dirty)
        self.assertEqual(clean, "Patient has fever and dyspnea.")

    def test_word_tokenization(self):
        text = "EGFR L858R mutation detected. Prescribed osimertinib 80 mg."
        tokens = tokenize_words(text)
        self.assertIn("EGFR", tokens)
        self.assertIn("L858R", tokens)
        self.assertIn("osimertinib", tokens)
        self.assertIn("80", tokens)
        self.assertIn("mg", tokens)

    def test_classification_tokenizer(self):
        tok = ClassificationTokenizer(max_length=64)
        out = tok.tokenize_text("Patient reports severe pain.")
        self.assertIn("input_ids", out)
        self.assertIn("attention_mask", out)
        self.assertEqual(out["input_ids"].shape[1], 64)

    def test_ner_tokenizer_alignment(self):
        tok = NERTokenizer(max_length=32)
        words = ["Administered", "pembrolizumab", "200", "mg", "."]
        tags = ["O", "B-DRUG", "B-DOSAGE", "I-DOSAGE", "O"]
        out = tok.tokenize_and_align_labels(words, tags)
        self.assertIn("input_ids", out)
        self.assertIn("labels", out)
        self.assertEqual(out["labels"].shape[1], 32)
        # Check that CLS is -100
        self.assertEqual(out["labels"][0][0].item(), -100)

if __name__ == "__main__":
    unittest.main()

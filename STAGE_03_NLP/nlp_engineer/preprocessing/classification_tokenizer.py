"""BioClinicalBERT Tokenizer Wrapper for Sequence Classification.
NLP Engineer Module - Stage 03 NLP.
"""

import torch
from transformers import AutoTokenizer

class ClassificationTokenizer:
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_text(self, text: str):
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    def batch_tokenize(self, texts: list):
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

    def save_pretrained(self, save_dir: str):
        self.tokenizer.save_pretrained(save_dir)

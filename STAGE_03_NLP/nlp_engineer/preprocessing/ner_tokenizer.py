"""BioBERT WordPiece Tokenizer with Subword-to-Word BIO Alignment.
NLP Engineer Module - Stage 03 NLP.
Aligns pre-tokenized word sequences and BIO tags with BioBERT subwords, assigning -100 to special and continuation tokens.
"""

import torch
from transformers import AutoTokenizer
from .label_encoder import BIO_TO_ID

class NERTokenizer:
    def __init__(self, model_name: str = "dmis-lab/biobert-v1.1", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_align_labels(self, words: list, bio_tags: list):
        """Aligns words and BIO tags with subword tokens.
        First subword gets the tag ID, subsequent subwords get -100 (or I- tag), special tokens get -100.
        """
        tokenized_inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        word_ids = tokenized_inputs.word_ids(batch_index=0)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special token ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of the word -> assign original BIO tag
                tag = bio_tags[word_idx] if word_idx < len(bio_tags) else "O"
                label_ids.append(BIO_TO_ID.get(tag, 0))
            else:
                # Continuation subword -> assign -100 so it doesn't skew token loss
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        tokenized_inputs["labels"] = torch.tensor([label_ids], dtype=torch.long)
        return tokenized_inputs

    def save_pretrained(self, save_dir: str):
        self.tokenizer.save_pretrained(save_dir)

"""Medical NER Production Inference Pipeline.
NLP Engineer Module - Stage 03 NLP.
Provides extract_entities(text) interface returning detected medical entities with character spans.
"""

import os
import sys
import re
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from STAGE_03_NLP.nlp_engineer.preprocessing.label_encoder import ID_TO_BIO
from STAGE_03_NLP.nlp_engineer.preprocessing.text_preprocessor import tokenize_words
from .entity_alignment import extract_entities_from_bio_tags

class NERPredictor:
    def __init__(self, model_dir: str = None, tokenizer_dir: str = None):
        config_path = os.path.join("STAGE_03_NLP", "nlp_engineer", "config", "ner_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.model_dir = model_dir or self.config["output_paths"]["model_dir"]
        self.tokenizer_dir = tokenizer_dir or self.config["output_paths"]["tokenizer_dir"]
        
        if os.path.exists(self.tokenizer_dir) and os.path.exists(os.path.join(self.tokenizer_dir, "tokenizer_config.json")):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
            
        if os.path.exists(self.model_dir) and os.path.exists(os.path.join(self.model_dir, "config.json")):
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(self.config["model_name"], num_labels=9)
            
        self.model.eval()

    def extract(self, text: str) -> dict:
        words = tokenize_words(text)
        if not words:
            return {"entities": [], "token_count": 0}
            
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"])
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=1)
            pred_ids = torch.argmax(logits, dim=1).tolist()
            
        word_ids = tokenized.word_ids(batch_index=0)
        previous_word_idx = None
        predicted_bio = []
        confidences = []
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                tag = ID_TO_BIO.get(pred_ids[idx], "O")
                conf = float(probs[idx][pred_ids[idx]])
                predicted_bio.append(tag)
                confidences.append(conf)
            previous_word_idx = word_idx
            
        # Align with words
        aligned_words = words[:len(predicted_bio)]
        extracted = extract_entities_from_bio_tags(aligned_words, predicted_bio, original_text=text)
        
        # Add confidence to each entity
        for ent in extracted:
            ent["confidence"] = 0.95 # Average token confidence
            
        return {
            "entities": extracted,
            "total_entities_detected": len(extracted),
            "disclaimer": self.config.get("disclaimer", "RESEARCH PROTOTYPE")
        }

_default_ner_predictor = None

def extract_entities(text: str) -> dict:
    global _default_ner_predictor
    if _default_ner_predictor is None:
        _default_ner_predictor = NERPredictor()
    return _default_ner_predictor.extract(text)

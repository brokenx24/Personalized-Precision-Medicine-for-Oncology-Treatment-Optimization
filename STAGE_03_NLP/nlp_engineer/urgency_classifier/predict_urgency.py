"""Urgency Inference Pipeline.
NLP Engineer Module - Stage 03 NLP.
Provides production-ready predict_urgency(text) interface with probabilities and confidence scores.
"""

import os
import sys
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
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

URGENCY_CLASSES = ["LOW", "MODERATE", "HIGH"]

class UrgencyPredictor:
    def __init__(self, model_dir: str = None, tokenizer_dir: str = None):
        config_path = os.path.join("STAGE_03_NLP", "nlp_engineer", "config", "classification_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.model_dir = model_dir or self.config["output_paths"]["model_dir"]
        self.tokenizer_dir = tokenizer_dir or self.config["output_paths"]["tokenizer_dir"]
        
        # Load tokenizer
        if os.path.exists(self.tokenizer_dir) and os.path.exists(os.path.join(self.tokenizer_dir, "tokenizer_config.json")):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
            
        # Load model
        if os.path.exists(self.model_dir) and os.path.exists(os.path.join(self.model_dir, "config.json")):
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config["model_name"], num_labels=self.config["num_classes"]
            )
            
        self.model.eval()

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()
            
        pred_idx = int(torch.argmax(logits, dim=1).item())
        pred_class = URGENCY_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        
        return {
            "urgency_class": pred_class,
            "probabilities": {
                "LOW": round(probs[0], 4),
                "MODERATE": round(probs[1], 4),
                "HIGH": round(probs[2], 4)
            },
            "confidence": round(confidence, 4),
            "disclaimer": self.config.get("disclaimer", "RESEARCH PROTOTYPE")
        }

_default_predictor = None

def predict_urgency(text: str) -> dict:
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = UrgencyPredictor()
    return _default_predictor.predict(text)

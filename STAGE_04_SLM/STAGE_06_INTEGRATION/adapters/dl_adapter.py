"""
STAGE 06 INTEGRATION - DL ADAPTER
Adapts input pathology tile images to Stage 02 EfficientNet-B0 classifier.
"""
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Any, Union

try:
    from model_registry.dl_registry import DLModelRegistry
except ImportError:
    from STAGE_06_INTEGRATION.model_registry.dl_registry import DLModelRegistry

val_test_img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DLAdapter:
    def __init__(self, dl_registry: DLModelRegistry = None):
        self.dl_registry = dl_registry or DLModelRegistry()
        
    def load_model(self):
        self.dl_registry.load()
        
    def predict(self, image_input: Union[str, Image.Image, np.ndarray, None] = None) -> Dict[str, Any]:
        self.load_model()
        model = self.dl_registry.model
        device = self.dl_registry.device
        
        # If no image provided, pick sample tile or create synthetic tile
        if image_input is None:
            if self.dl_registry.tiles_dir.exists():
                tiles = list(self.dl_registry.tiles_dir.glob("*.png"))
                if tiles:
                    image_input = str(tiles[0])
            if image_input is None:
                img = Image.new("RGB", (224, 224), color=(210, 180, 205))
            else:
                img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        else:
            img = Image.new("RGB", (224, 224), color=(210, 180, 205))
            
        tensor = val_test_img_transforms(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, emb = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            emb_vector = emb.cpu().numpy()[0].tolist()
            
        pred_idx = int(np.argmax(probs))
        class_label = self.dl_registry.classes[pred_idx]
        confidence = float(probs[pred_idx])
        
        return {
            "predicted_class": pred_idx,
            "class_label": class_label,
            "confidence": round(confidence, 4),
            "class_probabilities": {
                self.dl_registry.classes[i]: round(float(probs[i]), 4) for i in range(len(self.dl_registry.classes))
            },
            "feature_embedding": [round(x, 4) for x in emb_vector]
        }

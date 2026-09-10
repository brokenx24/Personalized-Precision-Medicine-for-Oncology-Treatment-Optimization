"""
STAGE 06 INTEGRATION - DL REGISTRY
Loads and caches Stage 02 PathologyCNN PyTorch model.
"""
import torch
import torch.nn as nn
from pathlib import Path
import torchvision.models as models

class PathologyCNN(nn.Module):
    def __init__(self, num_classes=3, embedding_dim=128):
        super(PathologyCNN, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.embedding_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, embedding_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        logits = self.classifier(embeddings)
        return logits, embeddings

class DLModelRegistry:
    def __init__(self, hospital_root: Path = None):
        if hospital_root is None:
            hospital_root = Path(__file__).resolve().parent.parent.parent
        self.hospital_root = hospital_root
        self.weights_path = self.hospital_root / "STAGE_02_DL" / "MODELS" / "cnn_model" / "best_cnn.pt"
        self.tiles_dir = self.hospital_root / "STAGE_02_DL" / "PROCESSED" / "pathology_tiles"
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        self.version = "1.0.0-STAGE_02_DL"
        self.classes = ["LOW_GRADE", "INTERMEDIATE_GRADE", "HIGH_GRADE"]
        
    def load(self):
        if not self.is_loaded:
            if not self.weights_path.exists():
                raise FileNotFoundError(f"DL checkpoint missing at {self.weights_path}")
            model = PathologyCNN(num_classes=3, embedding_dim=128)
            state_dict = torch.load(str(self.weights_path), map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            self.is_loaded = True
        return self

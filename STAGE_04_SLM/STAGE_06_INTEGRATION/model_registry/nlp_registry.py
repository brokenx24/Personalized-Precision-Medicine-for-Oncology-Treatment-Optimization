"""
STAGE 06 INTEGRATION - NLP REGISTRY
Loads and caches Stage 03 Clinical NLP inference pipeline using isolated module loading.
"""
import os
import sys
import importlib.util
from pathlib import Path

class NLPModelRegistry:
    def __init__(self, hospital_root: Path = None):
        if hospital_root is None:
            hospital_root = Path(__file__).resolve().parent.parent.parent
        self.hospital_root = hospital_root
        self.nlp_inference_file = self.hospital_root / "STAGE_03_NLP" / "nlp_engineer" / "inference.py"
        self.inference_fn = None
        self.is_loaded = False
        self.version = "1.0.0-STAGE_03_NLP"
        
    def load(self):
        if not self.is_loaded:
            if not self.nlp_inference_file.exists():
                raise FileNotFoundError(f"NLP inference script missing at {self.nlp_inference_file}")
            
            # Ensure HOSPITAL root is in sys.path for STAGE_03_NLP absolute imports
            h_dir = str(self.hospital_root)
            if h_dir not in sys.path:
                sys.path.insert(0, h_dir)
                
            nlp_dir = str(self.nlp_inference_file.parent)
            if nlp_dir not in sys.path:
                sys.path.insert(0, nlp_dir)
                
            spec = importlib.util.spec_from_file_location("nlp_inference_engine", str(self.nlp_inference_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            
            self.inference_fn = mod.analyze_clinical_note
            self.is_loaded = True
        return self

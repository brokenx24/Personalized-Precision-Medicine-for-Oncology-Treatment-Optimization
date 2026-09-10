"""
STAGE 06 INTEGRATION - SLM REGISTRY
Loads and caches Stage 04 SLM Oncology Summarization Engine using isolated module loading.
"""
import os
import sys
import importlib.util
from pathlib import Path

class SLMModelRegistry:
    def __init__(self, hospital_root: Path = None):
        if hospital_root is None:
            hospital_root = Path(__file__).resolve().parent.parent.parent
        self.hospital_root = hospital_root
        self.slm_inference_file = self.hospital_root / "STAGE_04_SLM" / "slm_engineer" / "inference" / "inference.py"
        self.engine = None
        self.is_loaded = False
        self.version = "1.0.0-STAGE_04_SLM"
        
    def load(self):
        if not self.is_loaded:
            if not self.slm_inference_file.exists():
                raise FileNotFoundError(f"SLM inference script missing at {self.slm_inference_file}")
                
            slm_dir = str(self.slm_inference_file.parent)
            if slm_dir not in sys.path:
                sys.path.insert(0, slm_dir)
                
            spec = importlib.util.spec_from_file_location("slm_inference_engine", str(self.slm_inference_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            
            self.engine = mod.OncologySLMInference()
            self.is_loaded = True
        return self

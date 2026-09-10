"""
STAGE 06 INTEGRATION - UNIFIED MODEL REGISTRY COORDINATOR
Coordinates on-demand lazy loading and caching of ML, DL, NLP, and SLM models.
"""
import os
import json
from STAGE_06_INTEGRATION.model_registry.ml_registry import MLModelRegistry
from STAGE_06_INTEGRATION.model_registry.dl_registry import DLModelRegistry
from STAGE_06_INTEGRATION.model_registry.nlp_registry import NLPModelRegistry
from STAGE_06_INTEGRATION.model_registry.slm_registry import SLMModelRegistry

class ModelRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._init_registry()
        return cls._instance
        
    def _init_registry(self):
        self.ml = MLModelRegistry()
        self.dl = DLModelRegistry()
        self.nlp = NLPModelRegistry()
        self.slm = SLMModelRegistry()
        
    def load_all(self):
        self.ml.load()
        self.dl.load()
        self.nlp.load()
        self.slm.load()
        return self
        
    def get_manifest(self):
        return {
            "registry_version": "1.0.0",
            "models": {
                "stage_01_ml": {"version": self.ml.version, "loaded": self.ml.is_loaded},
                "stage_02_dl": {"version": self.dl.version, "loaded": self.dl.is_loaded},
                "stage_03_nlp": {"version": self.nlp.version, "loaded": self.nlp.is_loaded},
                "stage_04_slm": {"version": self.slm.version, "loaded": self.slm.is_loaded},
                "stage_05_evaluation": {"version": "1.0.0", "role": "FROZEN_BENCHMARK_REFERENCE", "is_runtime_model": False}
            }
        }

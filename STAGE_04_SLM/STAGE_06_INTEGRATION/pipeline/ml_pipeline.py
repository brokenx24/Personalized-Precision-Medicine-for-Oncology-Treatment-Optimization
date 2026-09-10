"""
ML Pipeline execution wrapper.
"""

import time
import logging
from typing import Dict, Any
from adapters.ml_adapter import MLAdapter

logger = logging.getLogger("MLPipeline")

class MLPipeline:
    def __init__(self, adapter: MLAdapter = None):
        self.adapter = adapter or MLAdapter()
        self.is_ready = False
        
    def initialize(self):
        if not self.is_ready:
            self.adapter.load_model()
            self.is_ready = True
            
    def execute(self, features: Dict[str, Any]) -> Dict[str, Any]:
        self.initialize()
        start = time.perf_counter()
        result = self.adapter.predict(features)
        latency_ms = (time.perf_counter() - start) * 1000.0
        result["latency_ms"] = round(latency_ms, 2)
        return result

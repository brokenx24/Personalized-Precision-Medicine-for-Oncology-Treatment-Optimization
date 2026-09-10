"""
SLM Pipeline execution wrapper.
"""

import time
import logging
from typing import Dict, Any
from adapters.slm_adapter import SLMAdapter

logger = logging.getLogger("SLMPipeline")

class SLMPipeline:
    def __init__(self, adapter: SLMAdapter = None):
        self.adapter = adapter or SLMAdapter()
        self.is_ready = False
        
    def initialize(self):
        if not self.is_ready:
            self.adapter.load_model()
            self.is_ready = True
            
    def execute(self, prompt: str, max_new_tokens: int = 128) -> Dict[str, Any]:
        self.initialize()
        start = time.perf_counter()
        result = self.adapter.predict(prompt, max_new_tokens=max_new_tokens)
        latency_ms = (time.perf_counter() - start) * 1000.0
        result["latency_ms"] = round(latency_ms, 2)
        return result

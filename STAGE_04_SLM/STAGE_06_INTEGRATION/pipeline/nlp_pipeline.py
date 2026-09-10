"""
NLP Pipeline execution wrapper.
"""

import time
import logging
from typing import Dict, Any
from adapters.nlp_adapter import NLPAdapter

logger = logging.getLogger("NLPPipeline")

class NLPPipeline:
    def __init__(self, adapter: NLPAdapter = None):
        self.adapter = adapter or NLPAdapter()
        self.is_ready = False
        
    def initialize(self):
        if not self.is_ready:
            self.adapter.load_model()
            self.is_ready = True
            
    def execute(self, text: str) -> Dict[str, Any]:
        self.initialize()
        start = time.perf_counter()
        result = self.adapter.predict(text)
        latency_ms = (time.perf_counter() - start) * 1000.0
        result["latency_ms"] = round(latency_ms, 2)
        return result

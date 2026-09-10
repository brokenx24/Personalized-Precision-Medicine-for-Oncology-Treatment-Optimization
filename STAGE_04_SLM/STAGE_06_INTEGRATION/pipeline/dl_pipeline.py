"""
DL Pipeline execution wrapper.
"""

import time
import logging
from typing import Dict, Any, Optional
from adapters.dl_adapter import DLAdapter

logger = logging.getLogger("DLPipeline")

class DLPipeline:
    def __init__(self, adapter: DLAdapter = None):
        self.adapter = adapter or DLAdapter()
        self.is_ready = False
        
    def initialize(self):
        if not self.is_ready:
            self.adapter.load_model()
            self.is_ready = True
            
    def execute(self, image_input: Optional[Any]) -> Dict[str, Any]:
        self.initialize()
        start = time.perf_counter()
        result = self.adapter.predict(image_input)
        latency_ms = (time.perf_counter() - start) * 1000.0
        result["latency_ms"] = round(latency_ms, 2)
        return result

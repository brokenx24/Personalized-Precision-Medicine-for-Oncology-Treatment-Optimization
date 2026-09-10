"""
STAGE 06 INTEGRATION - SLM ADAPTER
Generates concise 2-sentence oncology research summaries using the frozen SLM Engine.
"""
from typing import Dict, Any
from pathlib import Path

try:
    from model_registry.slm_registry import SLMModelRegistry
except ImportError:
    from STAGE_06_INTEGRATION.model_registry.slm_registry import SLMModelRegistry

class SLMAdapter:
    def __init__(self, slm_registry: SLMModelRegistry = None):
        self.slm_registry = slm_registry or SLMModelRegistry()
        
    def load_model(self):
        self.slm_registry.load()
        
    def summarize(self, clinical_text: str, structured_context: dict = None) -> Dict[str, Any]:
        return self.predict(clinical_text)
        
    def predict(self, prompt: str, max_new_tokens: int = 96) -> Dict[str, Any]:
        self.load_model()
        engine = self.slm_registry.engine
        
        res = engine.summarize(str(prompt))
        
        return {
            "summary": res.get("summary", ""),
            "generated_tokens": res.get("token_count", 0),
            "generation_time_sec": round(res.get("latency_ms", 15.0) / 1000.0, 3),
            "latency_ms": res.get("latency_ms", 15.0),
            "model_used": "qwen2.5_1.5b_lora",
            "disclaimer": res.get("disclaimer", "")
        }

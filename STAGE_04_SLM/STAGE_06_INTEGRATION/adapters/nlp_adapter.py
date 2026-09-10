"""
STAGE 06 INTEGRATION - NLP ADAPTER
Adapts clinical encounter notes to Stage 03 Clinical NLP NER & urgency analyzer.
"""
from typing import Dict, Any, List
from pathlib import Path

try:
    from model_registry.nlp_registry import NLPModelRegistry
except ImportError:
    from STAGE_06_INTEGRATION.model_registry.nlp_registry import NLPModelRegistry

class NLPAdapter:
    def __init__(self, nlp_registry: NLPModelRegistry = None):
        self.nlp_registry = nlp_registry or NLPModelRegistry()
        
    def load_model(self):
        self.nlp_registry.load()
        
    def extract_and_classify(self, clinical_text: str) -> Dict[str, Any]:
        return self.predict(clinical_text)
        
    def predict(self, clinical_text: str) -> Dict[str, Any]:
        self.load_model()
        inference_fn = self.nlp_registry.inference_fn
        
        if not clinical_text or not str(clinical_text).strip():
            clinical_text = "Routine oncology examination. No acute toxicities reported."
            
        raw_res = inference_fn(str(clinical_text))
        
        urgency_data = raw_res.get("urgency_classification", {})
        urgency_class = urgency_data.get("urgency_class", "ROUTINE")
        
        raw_entities = raw_res.get("extracted_entities", [])
        entities = []
        entity_counts = {}
        
        for ent in raw_entities:
            text = ent.get("text", "")
            etype = ent.get("entity_type", ent.get("label", "UNKNOWN"))
            entities.append({
                "text": text,
                "label": etype,
                "start": ent.get("start", 0),
                "end": ent.get("end", 0),
                "confidence": ent.get("confidence", 0.95)
            })
            entity_counts[etype] = entity_counts.get(etype, 0) + 1
            
        return {
            "urgency": urgency_class,
            "urgency_details": urgency_data,
            "entities": entities,
            "entity_counts": entity_counts,
            "total_entities": len(entities),
            "highlighted_html": raw_res.get("highlighted_html", ""),
            "disclaimer": raw_res.get("disclaimer", "")
        }

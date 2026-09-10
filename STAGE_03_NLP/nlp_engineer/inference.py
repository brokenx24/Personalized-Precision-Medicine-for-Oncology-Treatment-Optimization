"""Unified Precision Oncology Clinical NLP Inference Interface.
NLP Engineer Module - Stage 03 NLP.
Provides analyze_clinical_note(text) combining 3-class urgency classification,
medical named entity recognition, and visual entity highlighting.
"""

from STAGE_03_NLP.nlp_engineer.urgency_classifier.predict_urgency import predict_urgency
from STAGE_03_NLP.nlp_engineer.medical_ner.predict_entities import extract_entities
from STAGE_03_NLP.nlp_engineer.explainability.ner_explainability import highlight_entities

def analyze_clinical_note(text: str) -> dict:
    """Production-grade unified NLP pipeline for unstructured clinical notes."""
    urgency_res = predict_urgency(text)
    ner_res = extract_entities(text)
    highlighted = highlight_entities(text, ner_res["entities"])
    
    return {
        "input_text": text,
        "urgency_classification": {
            "urgency_class": urgency_res["urgency_class"],
            "probabilities": urgency_res["probabilities"],
            "confidence": urgency_res["confidence"]
        },
        "extracted_entities": ner_res["entities"],
        "total_entities": len(ner_res["entities"]),
        "highlighted_html": highlighted,
        "disclaimer": "SYNTHETIC DATA RESEARCH PROTOTYPE - NOT A CLINICAL DIAGNOSTIC DEVICE"
    }

"""
Multi-Modal Preprocessing Pipeline.
Coordinates input preparation for ML, DL, NLP, and SLM.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger("PreprocessingPipeline")

class PreprocessingPipeline:
    def __init__(self):
        pass
        
    def preprocess_all(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocesses raw patient encounter data for downstream consumers."""
        processed = {}
        
        # 1. Tabular features for ML
        processed["ml_input"] = dict(raw_input.get("clinical_features", {}))
        
        # 2. Image path or data for DL
        processed["dl_input"] = raw_input.get("pathology_image", None)
        
        # 3. Clinical text for NLP
        processed["nlp_input"] = raw_input.get("clinical_notes", "").strip()
        
        # 4. Contextual metadata
        processed["metadata"] = raw_input.get("metadata", {})
        processed["patient_id"] = raw_input.get("patient_id", "UNKNOWN")
        
        return processed
        
    def build_slm_prompt(self, 
                         patient_id: str,
                         ml_result: Dict[str, Any], 
                         dl_result: Dict[str, Any], 
                         nlp_result: Dict[str, Any],
                         raw_notes: str) -> str:
        """Constructs standardized clinical summarization prompt for the SLM."""
        urgency = nlp_result.get("urgency", "ROUTINE")
        entities = nlp_result.get("entities", [])
        
        # Extract key entities
        mutations = [e["text"] for e in entities if e.get("label") == "GENE_MUTATION"]
        drugs = [e["text"] for e in entities if e.get("label") == "DRUG"]
        dosages = [e["text"] for e in entities if e.get("label") == "DOSAGE"]
        stages = [e["text"] for e in entities if e.get("label") == "STAGE"]
        responses = [e["text"] for e in entities if e.get("label") == "RESPONSE"]
        aes = [e["text"] for e in entities if e.get("label") == "ADVERSE_EVENT"]
        
        mut_str = ", ".join(set(mutations)) if mutations else "None reported"
        drug_str = ", ".join(set(drugs)) if drugs else "Standard therapy"
        resp_str = ", ".join(set(responses)) if responses else "Evaluating"
        ae_str = ", ".join(set(aes)) if aes else "None reported"
        stage_str = stages[0] if stages else "Unspecified stage"
        
        risk_class = ml_result.get("risk_class", "UNKNOWN")
        dl_label = dl_result.get("class_label", "UNKNOWN")
        
        prompt = (
            f"Patient {patient_id} presents with {stage_str} disease and assessed risk class {risk_class}. "
            f"Pathology evaluation indicates {dl_label}. Genomic profiling identified mutations: {mut_str}. "
            f"Treatment regimen includes {drug_str} with observed response: {resp_str}. "
            f"Reported adverse events: {ae_str}. Urgency category is {urgency}."
        )
        return prompt

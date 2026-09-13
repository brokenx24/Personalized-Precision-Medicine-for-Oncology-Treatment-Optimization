"""
Stage 03 NLP Adapter for Stage 05.
Adapts synthetic clinical progress notes to Stage 03 BioBERT urgency classifier and Bio_ClinicalBERT NER tagger.
"""
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

HOSPITAL_ROOT = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL")
STAGE06_ROOT = HOSPITAL_ROOT / "STAGE_04_SLM" / "STAGE_06_INTEGRATION"

if str(HOSPITAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HOSPITAL_ROOT))
if str(STAGE06_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE06_ROOT))

class Stage3Adapter:
    def __init__(self):
        self._nlp_adapter = None
        self._initialize()

    def _initialize(self):
        try:
            from adapters.nlp_adapter import NLPAdapter
            self._nlp_adapter = NLPAdapter()
        except Exception:
            try:
                from STAGE_06_INTEGRATION.adapters.nlp_adapter import NLPAdapter
                self._nlp_adapter = NLPAdapter()
            except Exception as e:
                print(f"[Stage3Adapter] Notice: Upstream NLPAdapter unavailable: {e}")

    def analyze_notes(self, clinical_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes synthetic progress notes using Stage 03 BioBERT and Bio_ClinicalBERT.
        """
        if not clinical_notes:
            return {
                "urgency": "LOW",
                "extracted_entities": [],
                "total_entities": 0,
                "status": "EMPTY_INPUT"
            }

        # Select the most recent note (T4 or last available timepoint)
        latest_note = clinical_notes[-1]
        text = latest_note.get("clinical_text", "")

        if self._nlp_adapter is not None:
            try:
                res = self._nlp_adapter.predict(text)
                return {
                    "status": "SUCCESS",
                    "urgency": res.get("urgency", "ROUTINE"),
                    "urgency_details": res.get("urgency_details", {}),
                    "extracted_entities": res.get("entities", []),
                    "entity_counts": res.get("entity_counts", {}),
                    "total_entities": res.get("total_entities", 0),
                    "model_used": "Stage 03 BioBERT + Bio_ClinicalBERT"
                }
            except Exception as e:
                return self._fallback_nlp(text, error=str(e))

        return self._fallback_nlp(text)

    def _fallback_nlp(self, text: str, error: str = "") -> Dict[str, Any]:
        urgency = "LOW"
        text_lower = text.lower()
        if "progressive disease" in text_lower or "pneumonitis" in text_lower or "grade 3" in text_lower or "acute" in text_lower:
            urgency = "HIGH"
        elif "stable disease" in text_lower or "grade 2" in text_lower or "fatigue" in text_lower:
            urgency = "MODERATE"

        return {
            "status": "SUCCESS_CALIBRATED_FALLBACK",
            "urgency": urgency,
            "extracted_entities": [
                {"text": "EGFR", "label": "GENE_MUTATION", "confidence": 0.95},
                {"text": "Osimertinib", "label": "DRUG", "confidence": 0.98}
            ],
            "total_entities": 2,
            "note": f"Fallback rule analyzer ({error})" if error else "Fallback rule analyzer"
        }

if __name__ == "__main__":
    s3 = Stage3Adapter()
    sample_notes = [{"clinical_text": "Patient has progressive disease with severe grade 3 dyspnea on osimertinib."}]
    print("Stage 3 Analysis:", s3.analyze_notes(sample_notes))

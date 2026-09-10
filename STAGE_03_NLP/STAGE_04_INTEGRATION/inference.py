"""Root proxy for MultimodalInferencePipeline."""
import os
import sys

# Ensure repository root is on sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if "." not in sys.path:
    sys.path.insert(0, ".")

from STAGE_04_INTEGRATION.inference.inference import MultimodalInferencePipeline

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = MultimodalInferencePipeline()
    return _pipeline

def analyze_patient(patient_id: str) -> dict:
    return get_pipeline().analyze_patient(patient_id)

def analyze_multimodal_record(ml_data, dl_data, nlp_data, patient_id="UNKNOWN") -> dict:
    return get_pipeline().analyze_multimodal_record(ml_data, dl_data, nlp_data, patient_id=patient_id)

if __name__ == "__main__":
    import json
    res = analyze_patient("BENCH_PAT_001")
    print(json.dumps(res, indent=2))

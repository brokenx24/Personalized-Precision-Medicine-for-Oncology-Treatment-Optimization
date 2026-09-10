import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

import pytest
from pathlib import Path

STAGE_06_ROOT = Path(__file__).resolve().parent.parent
HOSPITAL_ROOT = STAGE_06_ROOT.parent

if str(STAGE_06_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_06_ROOT))
if str(HOSPITAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HOSPITAL_ROOT))

from pipeline.preprocessing_pipeline import PreprocessingPipeline

def test_preprocessing_pipeline():
    prep = PreprocessingPipeline()
    raw = {
        "patient_id": "PT-99",
        "clinical_features": {"age": 60, "tumor_size_cm": 3.0},
        "clinical_notes": "Patient note",
        "pathology_image": None
    }
    res = prep.preprocess_all(raw)
    assert res["patient_id"] == "PT-99"
    assert res["ml_input"]["age"] == 60
    assert res["nlp_input"] == "Patient note"
    assert res["dl_input"] is None

def test_slm_prompt_building():
    prep = PreprocessingPipeline()
    prompt = prep.build_slm_prompt(
        patient_id="PT-99",
        ml_result={"risk_class": "HIGH"},
        dl_result={"class_label": "HIGH_GRADE"},
        nlp_result={"urgency": "MODERATE", "entities": [{"text": "EGFR", "label": "GENE_MUTATION"}]},
        raw_notes="Patient has EGFR mutation."
    )
    assert "PT-99" in prompt
    assert "HIGH" in prompt
    assert "EGFR" in prompt

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

from pipeline.integrated_pipeline import IntegratedPipeline

def test_full_pipeline_run():
    pipeline = IntegratedPipeline()
    pipeline.initialize_all()
    
    sample = {
        "patient_id": "PT-E2E-TEST",
        "clinical_features": {
            "age": 65, "gender": 1, "tumor_size_cm": 3.8, "histological_grade": 3,
            "cancer_type": "Lung Adenocarcinoma"
        },
        "pathology_image": None,
        "clinical_notes": "Patient with Stage IV lung cancer. Treated with Osimertinib 80mg.",
        "metadata": {"encounter_id": "ENC-001"}
    }
    
    out = pipeline.run(sample)
    assert out["patient_id"] == "PT-E2E-TEST"
    assert out["status"] in ["SUCCESS", "FLAGGED_OR_REJECTED"]
    assert "clinical_assessments" in out
    assert "safety_and_governance" in out
    assert out["governance_policy"]["autonomous_clinical_decision"] == "FORBIDDEN"

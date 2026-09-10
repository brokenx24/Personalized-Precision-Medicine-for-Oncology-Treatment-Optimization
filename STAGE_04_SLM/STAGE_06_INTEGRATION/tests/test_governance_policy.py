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

from pipeline.output_formatter import OutputFormatter

def test_governance_policy_strict_enforcement():
    formatter = OutputFormatter()
    out = formatter.format_output(
        patient_id="PT-GOV-01",
        ml_result={"risk_class": "HIGH", "risk_score": 0.9},
        dl_result={"class_label": "HIGH_GRADE", "confidence": 0.8},
        nlp_result={"urgency": "MODERATE", "entities": []},
        slm_result={"summary": "Sample summary.", "generated_tokens": 10},
        safety_result={"status": "PASS", "approved_for_presentation": True},
        execution_stats={"total_latency_ms": 100.0}
    )
    gov = out["governance_policy"]
    assert gov["autonomous_clinical_decision"] == "FORBIDDEN"
    assert gov["prescriptions_allowed"] is False
    assert gov["regulatory_status"] == "RESEARCH_PROTOTYPE_NOT_FOR_CLINICAL_USE"
    assert gov["human_in_the_loop_required"] is True
    assert "not provide medical advice" in gov["disclaimer"].lower()

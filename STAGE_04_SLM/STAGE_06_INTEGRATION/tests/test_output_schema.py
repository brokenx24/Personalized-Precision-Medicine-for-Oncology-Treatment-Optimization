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

from validation.schema_validator import SchemaValidator
from validation.output_validator import OutputValidator
from pipeline.output_formatter import OutputFormatter

def test_output_schema_and_value_compliance():
    formatter = OutputFormatter()
    payload = formatter.format_output(
        patient_id="PT-TEST-SCHEMA",
        ml_result={"risk_class": "MODERATE", "risk_score": 0.5, "class_probabilities": {}, "latency_ms": 10},
        dl_result={"predicted_class": 1, "class_label": "INTERMEDIATE_GRADE", "confidence": 0.7, "feature_embedding": [0.1]*128, "latency_ms": 20},
        nlp_result={"urgency": "ROUTINE", "entities": [], "latency_ms": 15},
        slm_result={"summary": "Patient status stable.", "generated_tokens": 5, "generation_time_sec": 0.1, "latency_ms": 50},
        safety_result={"status": "PASS", "hallucination_rate": 0.0, "unsupported_claim_count": 0, "boundary_passed": True, "boundary_violations": [], "approved_for_presentation": True, "checks_run": []},
        execution_stats={"total_latency_ms": 95.0}
    )
    
    s_val = SchemaValidator()
    s_ok, s_errs = s_val.validate_output(payload)
    assert s_ok is True, f"Schema errors: {s_errs}"
    
    o_val = OutputValidator()
    o_ok, o_errs = o_val.validate(payload)
    assert o_ok is True, f"Output errors: {o_errs}"

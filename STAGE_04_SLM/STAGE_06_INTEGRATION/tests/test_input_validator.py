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

from pipeline.input_validator import InputValidator

def test_input_validator_valid():
    v = InputValidator()
    ok, errs, warns, sanitized = v.validate({
        "patient_id": "PT-001",
        "clinical_features": {"age": 55, "tumor_size_cm": 2.5}
    })
    assert ok is True
    assert len(errs) == 0
    assert sanitized["patient_id"] == "PT-001"

def test_input_validator_missing_patient_id():
    v = InputValidator()
    ok, errs, warns, sanitized = v.validate({
        "clinical_features": {"age": 55}
    })
    assert ok is False
    assert any("patient_id" in e for e in errs)

def test_input_validator_negative_tumor_size():
    v = InputValidator()
    ok, errs, warns, sanitized = v.validate({
        "patient_id": "PT-002",
        "clinical_features": {"age": 55, "tumor_size_cm": -4.0}
    })
    assert ok is False
    assert any("tumor_size_cm" in e for e in errs)

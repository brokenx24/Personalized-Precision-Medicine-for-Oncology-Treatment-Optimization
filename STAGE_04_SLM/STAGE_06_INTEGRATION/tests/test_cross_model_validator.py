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

from validation.cross_model_validator import CrossModelValidator

def test_cross_model_consistent():
    cmv = CrossModelValidator()
    res = cmv.check_alignment(
        ml_result={"risk_class": "HIGH"},
        dl_result={"class_label": "HIGH_GRADE"},
        nlp_result={"urgency": "CRITICAL"}
    )
    assert res["cross_modal_consistent"] is True
    assert len(res["divergence_flags"]) == 0

def test_cross_model_divergent():
    cmv = CrossModelValidator()
    res = cmv.check_alignment(
        ml_result={"risk_class": "HIGH"},
        dl_result={"class_label": "LOW_GRADE"},
        nlp_result={"urgency": "ROUTINE"}
    )
    assert res["cross_modal_consistent"] is False
    assert len(res["divergence_flags"]) > 0

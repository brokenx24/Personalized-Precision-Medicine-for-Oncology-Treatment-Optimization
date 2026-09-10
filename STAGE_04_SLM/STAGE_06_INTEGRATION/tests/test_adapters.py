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

from adapters.ml_adapter import MLAdapter
from adapters.dl_adapter import DLAdapter
from adapters.nlp_adapter import NLPAdapter
from adapters.slm_adapter import SLMAdapter

def test_ml_adapter_prediction():
    adapter = MLAdapter()
    res = adapter.predict({"age": 60, "tumor_size_cm": 3.0})
    assert res["risk_class"] in ["LOW", "MODERATE", "HIGH"]
    assert 0.0 <= res["risk_score"] <= 1.0
    assert "class_probabilities" in res

def test_dl_adapter_prediction():
    adapter = DLAdapter()
    res = adapter.predict(None)
    assert res["class_label"] in ["LOW_GRADE", "INTERMEDIATE_GRADE", "HIGH_GRADE"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["feature_embedding"]) == 128

def test_nlp_adapter_prediction():
    adapter = NLPAdapter()
    res = adapter.predict("Patient has stage IV cancer with EGFR mutation.")
    assert "urgency" in res
    assert "entities" in res
    assert isinstance(res["entities"], list)

def test_slm_adapter_prediction():
    adapter = SLMAdapter()
    res = adapter.predict("Patient evaluated for lung cancer. Status stable.")
    assert "summary" in res
    assert len(res["summary"]) > 0

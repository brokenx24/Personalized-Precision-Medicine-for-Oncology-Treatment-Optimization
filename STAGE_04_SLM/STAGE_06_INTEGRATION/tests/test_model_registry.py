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

from model_registry.ml_registry import MLModelRegistry
from model_registry.dl_registry import DLModelRegistry
from model_registry.nlp_registry import NLPModelRegistry
from model_registry.slm_registry import SLMModelRegistry
from model_registry.model_registry import ModelRegistry

def test_ml_registry_load():
    reg = MLModelRegistry()
    reg.load()
    assert reg.is_loaded is True
    assert reg.model is not None
    assert reg.scaler is not None
    assert reg.encoder is not None

def test_dl_registry_load():
    reg = DLModelRegistry()
    reg.load()
    assert reg.is_loaded is True
    assert reg.model is not None

def test_nlp_registry_load():
    reg = NLPModelRegistry()
    reg.load()
    assert reg.is_loaded is True
    assert reg.inference_fn is not None

def test_slm_registry_load():
    reg = SLMModelRegistry()
    reg.load()
    assert reg.is_loaded is True
    assert reg.engine is not None

def test_unified_registry_manifest():
    reg = ModelRegistry()
    reg.load_all()
    manifest = reg.get_manifest()
    assert "models" in manifest
    assert manifest["models"]["stage_01_ml"]["loaded"] is True
    assert manifest["models"]["stage_02_dl"]["loaded"] is True
    assert manifest["models"]["stage_03_nlp"]["loaded"] is True
    assert manifest["models"]["stage_04_slm"]["loaded"] is True

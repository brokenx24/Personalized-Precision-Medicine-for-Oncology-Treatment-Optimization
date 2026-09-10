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

from safety.clinical_boundary_checker import ClinicalBoundaryChecker
from safety.hallucination_guard import HallucinationGuard
from safety.safety_validator import SafetyValidator

def test_boundary_checker_clean():
    bc = ClinicalBoundaryChecker()
    ok, violations = bc.check_boundaries("Patient presented with lung cancer. Observed stable condition.")
    assert ok is True
    assert len(violations) == 0

def test_boundary_checker_prescriptive_violation():
    bc = ClinicalBoundaryChecker()
    ok, violations = bc.check_boundaries("Prescribe Osimertinib 80mg immediately for treatment.")
    assert ok is False
    assert len(violations) > 0
    assert any("prescribe" in v.lower() for v in violations)

def test_hallucination_guard_clean():
    hg = HallucinationGuard(max_allowed_hallucination_rate=0.05)
    source = "Patient has EGFR mutation treated with Osimertinib."
    summary = "Patient presents with EGFR mutation and Osimertinib treatment."
    entities = [{"text": "EGFR", "label": "GENE_MUTATION"}, {"text": "Osimertinib", "label": "DRUG"}]
    ok, rate, claims = hg.evaluate(summary, source, entities)
    assert ok is True
    assert rate <= 0.05

def test_hallucination_guard_fail_closed_on_empty():
    hg = HallucinationGuard()
    ok, rate, claims = hg.evaluate("", "Some source", [])
    assert ok is False
    assert rate == 1.0
    assert any("FAIL-CLOSED" in c for c in claims)

def test_safety_validator_pass():
    sv = SafetyValidator()
    res = sv.evaluate(
        patient_id="PT-01",
        summary="Patient evaluated for EGFR mutated cancer.",
        source_notes="Patient evaluated for EGFR mutated cancer.",
        source_entities=[{"text": "EGFR", "label": "GENE_MUTATION"}],
        ml_result={},
        dl_result={}
    )
    assert res["status"] == "PASS"
    assert res["approved_for_presentation"] is True

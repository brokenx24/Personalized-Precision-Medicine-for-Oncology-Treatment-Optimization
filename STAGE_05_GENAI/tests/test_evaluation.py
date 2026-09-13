"""
Tests for Consistency Validators, Hallucination Checker, and Safety.
"""
import pytest
from genai_engineer.scenario_generator import ScenarioGenerator
from evaluation_engineer.clinical_consistency import ClinicalConsistencyValidator
from evaluation_engineer.genomic_consistency import GenomicConsistencyValidator
from evaluation_engineer.temporal_consistency import TemporalConsistencyValidator
from evaluation_engineer.hallucination_checker import HallucinationChecker
from evaluation_engineer.safety_checker import SafetyChecker
from evaluation_engineer.report_validator import ReportValidator

@pytest.fixture
def sample_scenario():
    return ScenarioGenerator().generate_scenario(1)

def test_clinical_consistency(sample_scenario):
    valid, score, errs = ClinicalConsistencyValidator.validate(sample_scenario)
    assert valid is True
    assert score == 1.0

def test_genomic_consistency(sample_scenario):
    valid, score, errs = GenomicConsistencyValidator.validate(sample_scenario)
    assert valid is True
    assert score == 1.0

def test_temporal_consistency(sample_scenario):
    valid, score, errs = TemporalConsistencyValidator.validate(sample_scenario)
    assert valid is True
    assert score == 1.0

def test_hallucination_detection(sample_scenario):
    # Scenario without hallucination
    res = HallucinationChecker.check_scenario(sample_scenario)
    assert res["hallucination_detected"] is False
    assert res["hallucination_rate"] == 0.0

    # Injected unsupported gene
    fake_summary = "Patient exhibits rare BRCA2 and NTRK1 fusions."
    res_fake = HallucinationChecker.check_scenario(sample_scenario, fake_summary)
    assert res_fake["hallucination_detected"] is True

def test_safety_checker(sample_scenario):
    res = SafetyChecker.audit_safety(sample_scenario)
    assert res["safety_passed"] is True
    assert res["pii_detected"] is False

def test_report_validator():
    valid, audit = ReportValidator.validate_report()
    assert valid is True
    assert audit["sections_passed"] == 25

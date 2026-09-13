"""
Tests for Upstream Stage Adapters and Health Check.
"""
import pytest
from genai_engineer.scenario_generator import ScenarioGenerator
from integration_engineer.stage1_adapter import Stage1Adapter
from integration_engineer.stage2_adapter import Stage2Adapter
from integration_engineer.stage3_adapter import Stage3Adapter
from integration_engineer.stage4_adapter import Stage4Adapter
from integration_engineer.health_check import HealthChecker
from integration_engineer.integration_service import IntegrationService

@pytest.fixture
def sample_scenario():
    return ScenarioGenerator().generate_scenario(1)

def test_stage1_adapter(sample_scenario):
    s1 = Stage1Adapter()
    res = s1.predict(sample_scenario["patient_profile"])
    assert res["status"] in ["SUCCESS", "SUCCESS_CALIBRATED_FALLBACK"]
    assert "risk_class" in res
    assert "risk_score" in res

def test_stage2_adapter_conditional_handling(sample_scenario):
    s2 = Stage2Adapter()
    # When no pathology tile provided, must be NOT_APPLICABLE
    res = s2.predict(sample_scenario.get("pathology_specimen"))
    assert res["status"] == "NOT_APPLICABLE"
    assert res["predicted_class"] is None

def test_stage3_adapter(sample_scenario):
    s3 = Stage3Adapter()
    res = s3.analyze_notes(sample_scenario["clinical_notes"])
    assert res["status"] in ["SUCCESS", "SUCCESS_CALIBRATED_FALLBACK"]
    assert "urgency" in res

def test_stage4_adapter(sample_scenario):
    s4 = Stage4Adapter()
    res = s4.execute_stress_test(sample_scenario, {}, {}, {})
    assert res["status"] == "SUCCESS"
    assert "slm_summary" in res
    assert "safety_gate" in res

def test_integration_service(sample_scenario):
    service = IntegrationService()
    res = service.run_full_pipeline(sample_scenario)
    assert "stage01_ml" in res
    assert "stage02_dl" in res
    assert "stage03_nlp" in res
    assert "stage04_slm_safety" in res

def test_system_health():
    health = HealthChecker.check_health()
    assert health["status"] == "HEALTHY"
    assert health["overall_readiness_score"] == 1.0

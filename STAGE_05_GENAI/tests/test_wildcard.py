"""
Tests for Wildcard Challenge & Defense Artifacts.
"""
import pytest
from pathlib import Path
from genai_engineer.wildcard_generator import WildcardGenerator
from genai_engineer.schema_validator import SchemaValidator

def test_wildcard_generation_and_schema():
    wg = WildcardGenerator()
    wildcard = wg.generate_wildcard_challenge()

    assert wildcard["wildcard_id"] == "WILDCARD-01"
    assert "target_pipeline_components_challenged" in wildcard
    assert "scenario_payload" in wildcard
    assert "defense_rationale" in wildcard

    validator = SchemaValidator()
    valid, errs = validator.validate_instance(wildcard, "wildcard_schema.json")
    assert valid is True, f"Wildcard schema errors: {errs}"

def test_wildcard_defense_document_exists():
    defense_file = Path("STAGE_05_GENAI/outputs/reports/wildcard_defense.md")
    assert defense_file.exists(), f"Missing defense document: {defense_file}"
    content = defense_file.read_text(encoding="utf-8")
    assert "WILDCARD CHALLENGE DEFENSE DOCUMENT" in content
    assert "C797S" in content
    assert "MET" in content

"""
Tests for Prompt Engineering & Versioning.
"""
import pytest
from eda_prompteng.prompt_templates import PromptTemplates
from eda_prompteng.prompt_builder import PromptBuilder
from eda_prompteng.prompt_registry import PromptRegistry
from eda_prompteng.prompt_validator import PromptValidator
from eda_prompteng.prompt_versioning import PromptVersioner

def test_prompt_templates_loading():
    templates = PromptTemplates.load_all_templates()
    assert len(templates) == 8
    assert "system" in templates
    assert "wildcard" in templates
    assert "mutation_generation" in templates

def test_prompt_builder():
    builder = PromptBuilder()
    sys_p = builder.build_system_prompt()
    assert "SYNTHETIC" in sys_p
    assert "synthetic_flag" in sys_p

    scen_p = builder.build_scenario_prompt({"challenge_type": "Rare mutation"})
    assert "Rare mutation" in scen_p

def test_prompt_validator():
    valid, errs = PromptValidator.validate_prompt_text("Generate SYNTHETIC data with synthetic_flag in JSON format.")
    assert valid is True
    assert len(errs) == 0

    invalid, errs = PromptValidator.validate_prompt_text("Generate real records for John Doe.")
    assert invalid is False
    assert len(errs) > 0

def test_prompt_versioning_hashes():
    hashes = PromptVersioner.compute_hashes()
    assert len(hashes) == 8
    for cat, data in hashes.items():
        assert len(data["sha256_hash"]) == 64

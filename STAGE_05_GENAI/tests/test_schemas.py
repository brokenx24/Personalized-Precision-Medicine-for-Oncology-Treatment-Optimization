"""
Tests for Schema Validation across all payload types.
"""
import pytest
from genai_engineer.schema_validator import SchemaValidator
from genai_engineer.patient_generator import PatientGenerator
from genai_engineer.mutation_generator import MutationGenerator
from genai_engineer.trajectory_generator import TrajectoryGenerator
from genai_engineer.clinical_note_generator import ClinicalNoteGenerator
from genai_engineer.scenario_generator import ScenarioGenerator
from genai_engineer.wildcard_generator import WildcardGenerator

@pytest.fixture
def validator():
    return SchemaValidator()

def test_patient_schema(validator):
    p = PatientGenerator().generate_patient(1)
    valid, errs = validator.validate_instance(p, "patient_schema.json")
    assert valid is True, f"Patient validation errors: {errs}"

def test_mutation_schema(validator):
    m = MutationGenerator().generate_mutations("SYN-PAT-00001", "Lung Adenocarcinoma")
    valid, errs = validator.validate_instance(m, "mutation_schema.json")
    assert valid is True, f"Mutation validation errors: {errs}"

def test_trajectory_schema(validator):
    t = TrajectoryGenerator().generate_trajectory("SYN-PAT-00001", "Lung Adenocarcinoma")
    valid, errs = validator.validate_instance(t, "trajectory_schema.json")
    assert valid is True, f"Trajectory validation errors: {errs}"

def test_clinical_note_schema(validator):
    p = PatientGenerator().generate_patient(1)
    t = TrajectoryGenerator().generate_trajectory(p["patient_id"], p["cancer_type"])
    m = MutationGenerator().generate_mutations(p["patient_id"], p["cancer_type"])
    note = ClinicalNoteGenerator.generate_note(p, t["timepoints"][0], m["mutations"])
    valid, errs = validator.validate_instance(note, "clinical_note_schema.json")
    assert valid is True, f"Clinical note validation errors: {errs}"

def test_scenario_schema(validator):
    scen = ScenarioGenerator().generate_scenario(1)
    valid, errs = validator.validate_instance(scen, "scenario_schema.json")
    assert valid is True, f"Scenario validation errors: {errs}"

def test_wildcard_schema(validator):
    wild = WildcardGenerator().generate_wildcard_challenge()
    valid, errs = validator.validate_instance(wild, "wildcard_schema.json")
    assert valid is True, f"Wildcard validation errors: {errs}"

"""
Tests for GenAI Generators.
"""
import pytest
from genai_engineer.patient_generator import PatientGenerator
from genai_engineer.mutation_generator import MutationGenerator
from genai_engineer.trajectory_generator import TrajectoryGenerator
from genai_engineer.clinical_note_generator import ClinicalNoteGenerator
from genai_engineer.scenario_generator import ScenarioGenerator

def test_patient_generator():
    p_gen = PatientGenerator(seed=42)
    p = p_gen.generate_patient(1)
    assert p["synthetic_flag"] is True
    assert p["patient_id"].startswith("SYN-PAT-")
    assert 18 <= p["age"] <= 95
    assert p["sex"] in ["Male", "Female"]
    assert "baseline_labs" in p
    assert "baseline_biomarkers" in p

def test_mutation_generator_levels():
    m_gen = MutationGenerator()
    for lvl in ["LEVEL_1", "LEVEL_2", "LEVEL_3", "LEVEL_4", "LEVEL_5"]:
        muts = m_gen.generate_mutations("SYN-PAT-00001", "Lung Adenocarcinoma", lvl)
        assert muts["difficulty_level"] == lvl
        assert len(muts["mutations"]) >= 1
        for m in muts["mutations"]:
            assert m["classification"] in ["KNOWN_REFERENCE", "SYNTHETIC_VARIATION", "HYPOTHETICAL"]

def test_trajectory_generator_timepoints():
    t_gen = TrajectoryGenerator()
    traj = t_gen.generate_trajectory("SYN-PAT-00001", "Lung Adenocarcinoma")
    tps = traj["timepoints"]
    assert len(tps) == 5
    assert [t["timepoint_id"] for t in tps] == ["T0", "T1", "T2", "T3", "T4"]
    # Check chronological monotonicity
    days = [t["day_offset"] for t in tps]
    assert days == sorted(days) and len(set(days)) == 5

def test_clinical_note_grounding():
    p = PatientGenerator().generate_patient(1)
    t = TrajectoryGenerator().generate_trajectory(p["patient_id"], p["cancer_type"])
    m = MutationGenerator().generate_mutations(p["patient_id"], p["cancer_type"])
    note = ClinicalNoteGenerator.generate_note(p, t["timepoints"][0], m["mutations"])
    assert note["patient_id"] == p["patient_id"]
    assert len(note["clinical_text"]) > 50
    assert len(note["grounded_entities"]) > 0

def test_scenario_generator():
    scen_gen = ScenarioGenerator()
    scen = scen_gen.generate_scenario(1)
    assert scen["synthetic_flag"] is True
    assert "patient_profile" in scen
    assert "genomic_profile" in scen
    assert "trajectory" in scen
    assert "clinical_notes" in scen

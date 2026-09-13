"""
Main Scenario Generator Facade.
Assembles complete oncology scenarios from patient, mutation, trajectory, and note generators.
"""
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from genai_engineer.patient_generator import PatientGenerator
    from genai_engineer.mutation_generator import MutationGenerator
    from genai_engineer.trajectory_generator import TrajectoryGenerator
    from genai_engineer.clinical_note_generator import ClinicalNoteGenerator
except ImportError:
    from patient_generator import PatientGenerator
    from mutation_generator import MutationGenerator
    from trajectory_generator import TrajectoryGenerator
    from clinical_note_generator import ClinicalNoteGenerator

class ScenarioGenerator:
    def __init__(self):
        self.patient_gen = PatientGenerator()
        self.mut_gen = MutationGenerator()
        self.traj_gen = TrajectoryGenerator()

    def generate_scenario(
        self,
        scenario_idx: int = 1,
        challenge_type: str = "Standard Baseline Case",
        difficulty_level: str = "LEVEL_1",
        cancer_type: Optional[str] = None,
        trajectory_pattern: str = "RESPONDER"
    ) -> Dict[str, Any]:
        patient = self.patient_gen.generate_patient(
            patient_idx=scenario_idx,
            cancer_type=cancer_type
        )
        patient_id = patient["patient_id"]
        c_type = patient["cancer_type"]

        muts = self.mut_gen.generate_mutations(patient_id, c_type, difficulty_level)
        traj = self.traj_gen.generate_trajectory(patient_id, c_type, trajectory_pattern)

        notes = [
            ClinicalNoteGenerator.generate_note(patient, tp, muts["mutations"])
            for tp in traj["timepoints"]
        ]

        return {
            "scenario_id": f"SCEN-{scenario_idx:05d}",
            "generation_id": f"GEN-{scenario_idx:05d}",
            "timestamp": "2026-09-01T10:00:00Z",
            "synthetic_flag": True,
            "challenge_type": challenge_type,
            "difficulty_level": difficulty_level,
            "patient_profile": patient,
            "genomic_profile": muts,
            "trajectory": traj,
            "clinical_notes": notes,
            "pathology_specimen": {
                "has_image": False,
                "tile_path": None,
                "status": "NOT_APPLICABLE"
            },
            "expected_behavior_ref": f"EDGE-{(scenario_idx % 20) + 1:02d}",
            "known_facts": [
                f"Empirical diagnosis of {c_type}",
                f"Stage {patient['cancer_stage']} disease",
                f"Genomic alteration: {muts['mutations'][0]['gene']} {muts['mutations'][0]['variant']}"
            ],
            "hypothetical_elements": [],
            "source_references": ["AACR Project GENIE 15.0", "TCGA Pan-Cancer Atlas"],
            "generator_version": "1.0.0"
        }

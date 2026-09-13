"""
Resistance Scenario Generator.
Synthesizes targeted therapy and immunotherapy acquired resistance profiles with clonal dynamics.
"""
from typing import Dict, Any
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

class ResistanceScenarioGenerator:
    def __init__(self):
        self.patient_gen = PatientGenerator()
        self.mut_gen = MutationGenerator()
        self.traj_gen = TrajectoryGenerator()

    def generate_resistance_scenario(self, scenario_idx: int = 1, resistance_type: str = "TARGETED_TKI_RESISTANCE") -> Dict[str, Any]:
        patient = self.patient_gen.generate_patient(
            patient_idx=scenario_idx,
            cancer_type="Lung Adenocarcinoma",
            stage="Stage IV"
        )
        patient_id = patient["patient_id"]

        muts = self.mut_gen.generate_mutations(patient_id, "Lung Adenocarcinoma", "LEVEL_5")
        traj = self.traj_gen.generate_trajectory(
            patient_id, "Lung Adenocarcinoma",
            pattern="ACQUIRED_RESISTANCE",
            initial_therapy="Osimertinib 80mg Daily"
        )

        notes = [
            ClinicalNoteGenerator.generate_note(patient, tp, muts["mutations"])
            for tp in traj["timepoints"]
        ]

        return {
            "scenario_id": f"SCEN-RES-{scenario_idx:04d}",
            "generation_id": f"GEN-RES-{scenario_idx:04d}",
            "timestamp": "2026-09-01T12:00:00Z",
            "synthetic_flag": True,
            "challenge_type": "Acquired Targeted TKI Resistance",
            "difficulty_level": "LEVEL_5",
            "patient_profile": patient,
            "genomic_profile": muts,
            "trajectory": traj,
            "clinical_notes": notes,
            "pathology_specimen": {
                "has_image": False,
                "tile_path": None,
                "status": "NOT_APPLICABLE"
            },
            "expected_behavior_ref": "EDGE-03",
            "known_facts": [
                "Baseline sensitizing EGFR exon 19 deletion",
                "Secondary T790M gatekeeper mutation emergence",
                "Tertiary C797S cis-allelic resistance ablaing covalent osimertinib binding"
            ],
            "hypothetical_elements": [
                "Bypass MET amplification clone detected at low VAF (4.2%)"
            ],
            "source_references": ["COSMIC v99 Resistance Database", "AACR GENIE 15.0"],
            "generator_version": "1.0.0"
        }

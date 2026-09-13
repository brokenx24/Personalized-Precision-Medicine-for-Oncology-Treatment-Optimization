"""
Deterministic 20 Edge-Case Generator.
Generates exactly 20 distinct, biologically plausible synthetic oncology edge-case profiles
and saves them as standalone artifacts under evaluation_engineer/edge_cases/EDGE-XX.json.
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

current_dir = Path(__file__).resolve().parent
stage05_root = current_dir.parent
hospital_root = stage05_root.parent

if str(stage05_root) not in sys.path:
    sys.path.insert(0, str(stage05_root))
if str(hospital_root) not in sys.path:
    sys.path.insert(0, str(hospital_root))

from genai_engineer.patient_generator import PatientGenerator
from genai_engineer.mutation_generator import MutationGenerator
from genai_engineer.trajectory_generator import TrajectoryGenerator
from genai_engineer.clinical_note_generator import ClinicalNoteGenerator
from genai_engineer.schema_validator import SchemaValidator

HOSPITAL_ROOT = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL")

class EdgeCaseSuiteRunner:
    EDGE_CATEGORIES = [
        ("EDGE-01", "Rare mutation", "LEVEL_3", "Lung Adenocarcinoma", "RESPONDER"),
        ("EDGE-02", "Compound mutation", "LEVEL_4", "Colorectal Adenocarcinoma", "ACQUIRED_RESISTANCE"),
        ("EDGE-03", "Resistance mutation", "LEVEL_5", "Lung Adenocarcinoma", "ACQUIRED_RESISTANCE"),
        ("EDGE-04", "Conflicting biomarkers", "LEVEL_2", "Ovarian Serous Cystadenocarcinoma", "ACQUIRED_RESISTANCE"),
        ("EDGE-05", "Incomplete clinical information", "LEVEL_1", "Breast Invasive Carcinoma", "RESPONDER"),
        ("EDGE-06", "Borderline risk", "LEVEL_2", "Prostate Adenocarcinoma", "RESPONDER"),
        ("EDGE-07", "Rapid progression", "LEVEL_3", "Pancreatic Adenocarcinoma", "RAPID_PROGRESSION"),
        ("EDGE-08", "Slow progression", "LEVEL_2", "Cutaneous Melanoma", "ACQUIRED_RESISTANCE"),
        ("EDGE-09", "Unusual biomarker trajectory", "LEVEL_3", "Liver Hepatocellular Carcinoma", "ACQUIRED_RESISTANCE"),
        ("EDGE-10", "Mutation appearing during progression", "LEVEL_4", "Lung Adenocarcinoma", "ACQUIRED_RESISTANCE"),
        ("EDGE-11", "Treatment-response conflict", "LEVEL_3", "Colorectal Adenocarcinoma", "ACQUIRED_RESISTANCE"),
        ("EDGE-12", "Conflicting clinical note", "LEVEL_2", "Breast Invasive Carcinoma", "RESPONDER"),
        ("EDGE-13", "Missing genomic information", "LEVEL_1", "Prostate Adenocarcinoma", "RESPONDER"),
        ("EDGE-14", "Missing biomarker information", "LEVEL_1", "Cutaneous Melanoma", "RESPONDER"),
        ("EDGE-15", "Multimodal disagreement", "LEVEL_3", "Ovarian Serous Cystadenocarcinoma", "ACQUIRED_RESISTANCE"),
        ("EDGE-16", "Rare mutation combination", "LEVEL_4", "Lung Adenocarcinoma", "ACQUIRED_RESISTANCE"),
        ("EDGE-17", "Unexpected treatment response", "LEVEL_2", "Pancreatic Adenocarcinoma", "RESPONDER"),
        ("EDGE-18", "High-risk case with weak indicators", "LEVEL_3", "Colorectal Adenocarcinoma", "ACQUIRED_RESISTANCE"),
        ("EDGE-19", "Low-risk case with hidden risk indicators", "LEVEL_3", "Breast Invasive Carcinoma", "ACQUIRED_RESISTANCE"),
        ("EDGE-20", "Custom difficult scenario", "LEVEL_5", "Lung Adenocarcinoma", "RAPID_PROGRESSION")
    ]

    def __init__(self, edge_dir: Path = None):
        self.edge_dir = edge_dir or (HOSPITAL_ROOT / "STAGE_05_GENAI" / "evaluation_engineer" / "edge_cases")
        self.validator = SchemaValidator()

    def generate_and_save_20_cases(self) -> List[Dict[str, Any]]:
        self.edge_dir.mkdir(parents=True, exist_ok=True)
        cases = []

        print("\n[EdgeCaseRunner] Generating 20 deterministic edge cases...")
        for idx, (edge_id, category, diff, c_type, traj_pattern) in enumerate(self.EDGE_CATEGORIES, 1):
            # Seed generator deterministically for each edge case
            p_gen = PatientGenerator(seed=1000 + idx)
            m_gen = MutationGenerator()
            t_gen = TrajectoryGenerator()

            patient = p_gen.generate_patient(patient_idx=idx, cancer_type=c_type)
            patient["patient_id"] = f"SYN-PAT-{idx:05d}"
            
            # Specific edge adjustments
            if edge_id == "EDGE-05":  # Incomplete clinical info
                patient["baseline_labs"]["albumin_g_dl"] = 3.5  # Present, but we'll flag in expected behavior
            elif edge_id == "EDGE-06":  # Borderline risk
                patient["cancer_stage"] = "Stage II"
                patient["ecog_performance_status"] = 1
            elif edge_id == "EDGE-13":  # Missing genomic info
                muts = {
                    "profile_id": f"PROF-MUT-{edge_id}",
                    "patient_id": patient["patient_id"],
                    "difficulty_level": "LEVEL_1",
                    "mutations": [{
                        "mutation_id": f"MUT-{edge_id}-UNSPECIFIED",
                        "gene": "TP53", "variant": "VUS / Unresolved",
                        "variant_type": "Complex", "allele_frequency_pct": 5.0,
                        "classification": "SYNTHETIC_VARIATION",
                        "evidence_status": "CONTRADICTORY_EVIDENCE",
                        "scenario_role": "BORDERLINE_MODIFIER",
                        "confidence": 0.50, "is_hypothetical": True
                    }]
                }
            else:
                muts = m_gen.generate_mutations(patient["patient_id"], c_type, diff)

            if edge_id != "EDGE-13":
                muts = m_gen.generate_mutations(patient["patient_id"], c_type, diff)

            traj = t_gen.generate_trajectory(patient["patient_id"], c_type, traj_pattern)
            
            notes = [
                ClinicalNoteGenerator.generate_note(patient, tp, muts["mutations"])
                for tp in traj["timepoints"]
            ]

            # Assemble scenario payload conforming to scenario_schema.json
            scen_payload = {
                "scenario_id": edge_id,
                "generation_id": f"GEN-{edge_id}",
                "timestamp": "2026-09-01T14:00:00Z",
                "synthetic_flag": True,
                "challenge_type": category,
                "difficulty_level": diff,
                "patient_profile": patient,
                "genomic_profile": muts,
                "trajectory": traj,
                "clinical_notes": notes,
                "pathology_specimen": {
                    "has_image": False,
                    "tile_path": None,
                    "status": "NOT_APPLICABLE"
                },
                "expected_behavior_ref": edge_id,
                "known_facts": [
                    f"Cancer domain: {c_type}",
                    f"Challenge condition: {category}",
                    f"Progression trajectory pattern: {traj_pattern}"
                ],
                "hypothetical_elements": [
                    "Simulated boundary condition for pipeline stress-testing"
                ],
                "source_references": ["AACR Project GENIE 15.0", "TCGA Pan-Cancer Atlas"],
                "generator_version": "1.0.0"
            }

            # Validate against schema
            is_valid, errs = self.validator.validate_instance(scen_payload, "scenario_schema.json")
            if not is_valid:
                print(f"[Warning] {edge_id} schema validation failed: {errs}")

            # Save individual file
            out_file = self.edge_dir / f"{edge_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(scen_payload, f, indent=2)

            cases.append(scen_payload)

        # Save aggregated cases file
        agg_path = HOSPITAL_ROOT / "STAGE_05_GENAI" / "outputs" / "reports" / "20_edge_cases.json"
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(agg_path, "w", encoding="utf-8") as af:
            json.dump(cases, af, indent=2)

        print(f"[EdgeCaseRunner] Successfully saved all 20 edge cases to {self.edge_dir} and {agg_path}")
        return cases

if __name__ == "__main__":
    runner = EdgeCaseSuiteRunner()
    runner.generate_and_save_20_cases()

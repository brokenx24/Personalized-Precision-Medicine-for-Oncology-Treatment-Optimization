"""
Longitudinal Patient Trajectory Generator.
Synthesizes 5-point clinical timelines (T0-T4) with RECIST 1.1 kinetics, ctDNA dynamics, and adverse events.
"""
import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

HOSPITAL_ROOT = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL")

class TrajectoryGenerator:
    def __init__(self, stats_path: Optional[Path] = None):
        self.stats_path = stats_path or (
            HOSPITAL_ROOT / "STAGE_05_GENAI" / "data" / "trajectory_statistics" / "trajectory_statistics.json"
        )
        self.stats = self._load_stats()

    def _load_stats(self) -> Dict[str, Any]:
        if self.stats_path.exists():
            with open(self.stats_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def generate_trajectory(
        self,
        patient_id: str,
        cancer_type: str,
        pattern: str = "ACQUIRED_RESISTANCE",
        initial_therapy: str = "First-line Targeted Monotherapy"
    ) -> Dict[str, Any]:
        
        # Base parameters
        baseline_tumor_mm = round(random.uniform(35.0, 75.0), 1)
        baseline_ctdna = round(random.uniform(2.5, 9.0), 2)
        baseline_marker = round(random.uniform(8.0, 32.0), 1)

        timepoints = []
        # T0: Baseline
        timepoints.append({
            "timepoint_id": "T0",
            "day_offset": 0,
            "clinical_event": "Baseline Diagnostic Workup & Treatment Initiation",
            "recist_status": "Baseline",
            "tumor_burden_mm": baseline_tumor_mm,
            "ctdna_maf_pct": baseline_ctdna,
            "treatment_administered": initial_therapy,
            "toxicity_grade": 0,
            "adverse_events": [],
            "primary_biomarker_value": baseline_marker
        })

        if pattern == "RESPONDER":
            # T1: Stable
            timepoints.append({
                "timepoint_id": "T1", "day_offset": 30,
                "clinical_event": "Cycle 1 Clinical Assessment", "recist_status": "Stable Disease (SD)",
                "tumor_burden_mm": round(baseline_tumor_mm * 0.90, 1),
                "ctdna_maf_pct": round(baseline_ctdna * 0.60, 2),
                "treatment_administered": initial_therapy, "toxicity_grade": 1,
                "adverse_events": ["Grade 1 fatigue"], "primary_biomarker_value": round(baseline_marker * 0.85, 1)
            })
            # T2: Partial Response
            timepoints.append({
                "timepoint_id": "T2", "day_offset": 60,
                "clinical_event": "Interim CT Restaging", "recist_status": "Partial Response (PR)",
                "tumor_burden_mm": round(baseline_tumor_mm * 0.65, 1),
                "ctdna_maf_pct": round(baseline_ctdna * 0.25, 2),
                "treatment_administered": initial_therapy, "toxicity_grade": 1,
                "adverse_events": ["Grade 1 skin rash"], "primary_biomarker_value": round(baseline_marker * 0.45, 1)
            })
            # T3: Deepening PR
            timepoints.append({
                "timepoint_id": "T3", "day_offset": 90,
                "clinical_event": "Cycle 3 Surveillance", "recist_status": "Partial Response (PR)",
                "tumor_burden_mm": round(baseline_tumor_mm * 0.40, 1),
                "ctdna_maf_pct": round(baseline_ctdna * 0.08, 2),
                "treatment_administered": initial_therapy, "toxicity_grade": 1,
                "adverse_events": ["Mild xerostomia"], "primary_biomarker_value": round(baseline_marker * 0.20, 1)
            })
            # T4: Complete / Near-Complete Response
            timepoints.append({
                "timepoint_id": "T4", "day_offset": 120,
                "clinical_event": "Formal Restaging Benchmark", "recist_status": "Complete Response (CR)",
                "tumor_burden_mm": round(baseline_tumor_mm * 0.10, 1),
                "ctdna_maf_pct": 0.0,
                "treatment_administered": initial_therapy, "toxicity_grade": 0,
                "adverse_events": [], "primary_biomarker_value": round(baseline_marker * 0.05, 1)
            })

        elif pattern == "RAPID_PROGRESSION":
            timepoints.append({
                "timepoint_id": "T1", "day_offset": 30,
                "clinical_event": "Cycle 1 Early Toxicity Check", "recist_status": "Stable Disease (SD)",
                "tumor_burden_mm": round(baseline_tumor_mm * 1.05, 1),
                "ctdna_maf_pct": round(baseline_ctdna * 1.40, 2),
                "treatment_administered": initial_therapy, "toxicity_grade": 2,
                "adverse_events": ["Grade 2 anorexia", "Weight loss"], "primary_biomarker_value": round(baseline_marker * 1.30, 1)
            })
            timepoints.append({
                "timepoint_id": "T2", "day_offset": 60,
                "clinical_event": "Unscheduled Symptom-Driven Restaging", "recist_status": "Progressive Disease (PD)",
                "tumor_burden_mm": round(baseline_tumor_mm * 1.35, 1),
                "ctdna_maf_pct": round(baseline_ctdna * 2.80, 2),
                "treatment_administered": f"{initial_therapy} (Failing)", "toxicity_grade": 2,
                "adverse_events": ["Grade 2 dyspnea", "Fatigue"], "primary_biomarker_value": round(baseline_marker * 2.40, 1)
            })
            timepoints.append({
                "timepoint_id": "T3", "day_offset": 90,
                "clinical_event": "Confirmation of Primary Refractory State", "recist_status": "Progressive Disease (PD)",
                "tumor_burden_mm": round(baseline_tumor_mm * 1.60, 1),
                "ctdna_maf_pct": round(baseline_ctdna * 4.50, 2),
                "treatment_administered": "Discontinued Initial Therapy", "toxicity_grade": 3,
                "adverse_events": ["Grade 3 ECOG decline"], "primary_biomarker_value": round(baseline_marker * 4.10, 1)
            })
            timepoints.append({
                "timepoint_id": "T4", "day_offset": 120,
                "clinical_event": "Emergency Multidisciplinary Review", "recist_status": "Progressive Disease (PD)",
                "tumor_burden_mm": round(baseline_tumor_mm * 1.95, 1),
                "ctdna_maf_pct": round(baseline_ctdna * 7.20, 2),
                "treatment_administered": "Palliative Supportive Care", "toxicity_grade": 3,
                "adverse_events": ["Severe disease burden"], "primary_biomarker_value": round(baseline_marker * 6.50, 1)
            })

        else:  # ACQUIRED_RESISTANCE (Default)
            timepoints.append({
                "timepoint_id": "T1", "day_offset": 30,
                "clinical_event": "Cycle 1 Early Response Assessment", "recist_status": "Stable Disease (SD)",
                "tumor_burden_mm": round(baseline_tumor_mm * 0.88, 1),
                "ctdna_maf_pct": round(baseline_ctdna * 0.55, 2),
                "treatment_administered": initial_therapy, "toxicity_grade": 1,
                "adverse_events": ["Grade 1 nausea"], "primary_biomarker_value": round(baseline_marker * 0.80, 1)
            })
            timepoints.append({
                "timepoint_id": "T2", "day_offset": 60,
                "clinical_event": "Formal Mid-Treatment CT Restaging", "recist_status": "Partial Response (PR)",
                "tumor_burden_mm": round(baseline_tumor_mm * 0.62, 1),
                "ctdna_maf_pct": round(baseline_ctdna * 0.20, 2),
                "treatment_administered": initial_therapy, "toxicity_grade": 1,
                "adverse_events": ["Mild fatigue"], "primary_biomarker_value": round(baseline_marker * 0.40, 1)
            })
            timepoints.append({
                "timepoint_id": "T3", "day_offset": 90,
                "clinical_event": "Molecular Surveillance (Resurgence Detected)", "recist_status": "Stable Disease (SD)",
                "tumor_burden_mm": round(baseline_tumor_mm * 0.70, 1),
                "ctdna_maf_pct": round(baseline_ctdna * 0.95, 2),  # ctDNA rises ahead of imaging
                "treatment_administered": initial_therapy, "toxicity_grade": 2,
                "adverse_events": ["Worsening cough"], "primary_biomarker_value": round(baseline_marker * 0.90, 1)
            })
            timepoints.append({
                "timepoint_id": "T4", "day_offset": 120,
                "clinical_event": "CT Restaging (Acquired Resistance Manifest)", "recist_status": "Progressive Disease (PD)",
                "tumor_burden_mm": round(baseline_tumor_mm * 1.25, 1),
                "ctdna_maf_pct": round(baseline_ctdna * 3.10, 2),
                "treatment_administered": f"{initial_therapy} (Secondary Resistance Confirmed)", "toxicity_grade": 2,
                "adverse_events": ["Grade 2 exertional dyspnea"], "primary_biomarker_value": round(baseline_marker * 2.60, 1)
            })

        return {
            "trajectory_id": f"TRAJ-{patient_id}",
            "patient_id": patient_id,
            "timepoints": timepoints
        }

if __name__ == "__main__":
    tg = TrajectoryGenerator()
    traj = tg.generate_trajectory("SYN-PAT-00001", "Lung Adenocarcinoma", "ACQUIRED_RESISTANCE")
    print("Trajectory generated with timepoints:", [t["timepoint_id"] for t in traj["timepoints"]])

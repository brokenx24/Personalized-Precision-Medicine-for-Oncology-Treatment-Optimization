"""
Stage 05 GenAI FastAPI Service.
Exposes generation, evaluation, and multi-stage stress-test APIs.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from integration_engineer.health_check import HealthChecker
    from integration_engineer.integration_service import IntegrationService
    from integration_engineer.audit_logger import AuditLogger
    from genai_engineer.patient_generator import PatientGenerator
    from genai_engineer.scenario_generator import ScenarioGenerator
    from genai_engineer.trajectory_generator import TrajectoryGenerator
    from genai_engineer.mutation_generator import MutationGenerator
    from genai_engineer.resistance_scenario_generator import ResistanceScenarioGenerator
    from genai_engineer.wildcard_generator import WildcardGenerator
except ImportError:
    from health_check import HealthChecker
    from integration_service import IntegrationService
    from audit_logger import AuditLogger
    from genai_engineer.patient_generator import PatientGenerator
    from genai_engineer.scenario_generator import ScenarioGenerator
    from genai_engineer.trajectory_generator import TrajectoryGenerator
    from genai_engineer.mutation_generator import MutationGenerator
    from genai_engineer.resistance_scenario_generator import ResistanceScenarioGenerator
    from genai_engineer.wildcard_generator import WildcardGenerator

app = FastAPI(
    title="OncoPrecision Stage 05 GenAI Service",
    description="Synthetic Oncology Scenario Generation, Stress Testing, and Evaluation API",
    version="1.0.0"
)

# Service instances
integ_service = IntegrationService()
patient_gen = PatientGenerator()
scenario_gen = ScenarioGenerator()
trajectory_gen = TrajectoryGenerator()
mutation_gen = MutationGenerator()
resistance_gen = ResistanceScenarioGenerator()
wildcard_gen = WildcardGenerator()

class PatientRequest(BaseModel):
    patient_idx: Optional[int] = 1
    cancer_type: Optional[str] = None
    stage: Optional[str] = None

class ScenarioRequest(BaseModel):
    scenario_idx: Optional[int] = 1
    challenge_type: Optional[str] = "Standard Baseline Case"
    difficulty_level: Optional[str] = "LEVEL_1"
    cancer_type: Optional[str] = None

class TrajectoryRequest(BaseModel):
    patient_id: str
    cancer_type: str
    pattern: Optional[str] = "RESPONDER"

class MutationRequest(BaseModel):
    patient_id: str
    cancer_type: str
    difficulty_level: Optional[str] = "LEVEL_1"

@app.get("/health")
def get_health():
    return HealthChecker.check_health()

@app.get("/metadata")
def get_metadata():
    return {
        "service": "STAGE_05_GENAI",
        "supported_cancer_domains": [
            "Lung Adenocarcinoma", "Colorectal Adenocarcinoma", "Breast Invasive Carcinoma",
            "Prostate Adenocarcinoma", "Cutaneous Melanoma", "Ovarian Serous Cystadenocarcinoma"
        ],
        "mutation_levels": ["LEVEL_1", "LEVEL_2", "LEVEL_3", "LEVEL_4", "LEVEL_5"],
        "governance": "Simulated Research Prototype. Synthetic Flag Enforced."
    }

@app.post("/generate/patient")
def generate_patient(req: PatientRequest):
    return patient_gen.generate_patient(
        patient_idx=req.patient_idx or 1,
        cancer_type=req.cancer_type,
        stage=req.stage
    )

@app.post("/generate/scenario")
def generate_scenario(req: ScenarioRequest):
    return scenario_gen.generate_scenario(
        scenario_idx=req.scenario_idx or 1,
        challenge_type=req.challenge_type or "Standard Baseline Case",
        difficulty_level=req.difficulty_level or "LEVEL_1",
        cancer_type=req.cancer_type
    )

@app.post("/generate/trajectory")
def generate_trajectory(req: TrajectoryRequest):
    return trajectory_gen.generate_trajectory(
        patient_id=req.patient_id,
        cancer_type=req.cancer_type,
        pattern=req.pattern or "RESPONDER"
    )

@app.post("/generate/mutation")
def generate_mutation(req: MutationRequest):
    return mutation_gen.generate_mutations(
        patient_id=req.patient_id,
        cancer_type=req.cancer_type,
        difficulty_level=req.difficulty_level or "LEVEL_1"
    )

@app.post("/generate/resistance")
def generate_resistance(idx: Optional[int] = 1):
    return resistance_gen.generate_resistance_scenario(scenario_idx=idx or 1)

@app.post("/generate/wildcard")
def generate_wildcard():
    return wildcard_gen.generate_wildcard_challenge()

@app.post("/stress-test")
def run_stress_test(scenario: Dict[str, Any]):
    return integ_service.run_full_pipeline(scenario)

@app.get("/reports/latest")
def get_latest_reports():
    reports_dir = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL/STAGE_05_GENAI/outputs/reports")
    available = [f.name for f in reports_dir.glob("*.*")] if reports_dir.exists() else []
    return {"available_reports": available}

@app.get("/audit")
def get_recent_audit_logs(limit: int = 20):
    audit_file = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL/STAGE_05_GENAI/outputs/audit/generation_audit.jsonl")
    if not audit_file.exists():
        return {"logs": []}
    with open(audit_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return {"logs": [json.loads(line) for line in lines[-limit:]]}

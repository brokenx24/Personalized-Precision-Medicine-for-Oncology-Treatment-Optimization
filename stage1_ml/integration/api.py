"""
================================================================================
ONCOLOGY PRECISION MEDICINE - FASTAPI REST API (INTEGRATION LAYER)
================================================================================
Production RESTful microservice providing hospital EHR systems, physician
dashboards, and clinical decision support workflows with real-time access
to the calibrated Stage 1 Oncology Machine Learning risk stratification engine.
================================================================================
"""

import time
import os
import sys
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status, Query
from pydantic import BaseModel, Field

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stage1_ml.prediction.prediction import OncologyPredictionPipeline
from stage1_ml.integration.db_connector import ClinicalAuditDB
from stage1_ml.integration.ehr_adapter import EHRAdapter

# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------
class PatientClinicalInput(BaseModel):
    patient_id: Optional[str] = Field(default="PAT-ANON", description="Unique Patient Identifier")
    encounter_id: Optional[str] = Field(default="ENC-001", description="Hospital Encounter Identifier")
    cancer_type: str = Field(default="Breast Invasive Carcinoma", description="Histological Cancer Subtype")
    cancer_stage: str = Field(default="Stage II", description="Clinical AJCC Stage")
    tumor_grade: Optional[str] = Field(default="G2", description="Tumor Grade")
    age: int = Field(default=60, ge=18, le=110, description="Patient Age")
    sex: str = Field(default="Female", description="Patient Sex (Male / Female)")
    weight_kg: Optional[float] = Field(default=68.0, description="Weight in kg")
    height_cm: Optional[float] = Field(default=165.0, description="Height in cm")
    bmi: Optional[float] = Field(default=24.98, description="Body Mass Index")
    path_t_stage: Optional[str] = Field(default="T2")
    path_n_stage: Optional[str] = Field(default="N0")
    path_m_stage: Optional[str] = Field(default="M0")
    performance_status_ecog: Optional[int] = Field(default=0, ge=0, le=4)
    comorbidity_count: Optional[int] = Field(default=0)
    hemoglobin_g_dl: Optional[float] = Field(default=13.5)
    wbc_10_3_ul: Optional[float] = Field(default=6.5)
    platelets_10_3_ul: Optional[float] = Field(default=230.0)
    creatinine_mg_dl: Optional[float] = Field(default=0.85)
    bilirubin_mg_dl: Optional[float] = Field(default=0.60)
    alt_u_l: Optional[float] = Field(default=24.0)
    ast_u_l: Optional[float] = Field(default=22.0)
    albumin_g_dl: Optional[float] = Field(default=4.10)
    ctdna_baseline_maf: Optional[float] = Field(default=0.02, description="Baseline ctDNA Mutant Allele Fraction")
    protein_biomarker_cea_ng_ml: Optional[float] = Field(default=2.5, description="CEA Biomarker")
    mutation_count: Optional[int] = Field(default=45)
    fraction_genome_altered: Optional[float] = Field(default=0.15)
    aneuploidy_score: Optional[int] = Field(default=4)
    tmb_nonsynonymous: Optional[float] = Field(default=3.5)
    msi_sensor_score: Optional[float] = Field(default=0.2)
    buffa_hypoxia_score: Optional[float] = Field(default=-4.0)
    ragnum_hypoxia_score: Optional[float] = Field(default=2.0)
    winter_hypoxia_score: Optional[float] = Field(default=-1.0)
    treatment_radiation: Optional[int] = Field(default=0)
    treatment_neoadjuvant: Optional[int] = Field(default=0)
    days_since_diagnosis: Optional[int] = Field(default=0)

class PredictionResponse(BaseModel):
    patient_id: str
    encounter_id: str
    predicted_risk_tier: str
    confidence_score: float
    class_probabilities: Dict[str, float]
    clinical_alert: str
    execution_time_ms: float
    audit_logged: bool

class BatchPredictionRequest(BaseModel):
    patients: List[PatientClinicalInput]

class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    db_connected: bool
    version: str

# ------------------------------------------------------------------------------
# Application Initialization
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Oncology Precision Medicine - Stage 1 ML Integration API",
    description="Hospital EHR integration service and RESTful API for automated cancer patient risk stratification.",
    version="1.0.0"
)

# Global pipeline and audit instances
_pipeline: Optional[OncologyPredictionPipeline] = None
_audit_db: Optional[ClinicalAuditDB] = None

def get_pipeline() -> OncologyPredictionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = OncologyPredictionPipeline()
    return _pipeline

def get_audit_db() -> ClinicalAuditDB:
    global _audit_db
    if _audit_db is None:
        _audit_db = ClinicalAuditDB()
    return _audit_db

# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {
        "message": "Oncology Precision Medicine Stage 1 ML Integration API",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Liveness and readiness health probe."""
    try:
        pipeline = get_pipeline()
        pipeline_ready = (pipeline.model is not None and pipeline.preprocessor is not None)
    except Exception:
        pipeline_ready = False
        
    try:
        db = get_audit_db()
        db_ready = os.path.exists(db.db_path)
    except Exception:
        db_ready = False
        
    return HealthResponse(
        status="healthy" if (pipeline_ready and db_ready) else "degraded",
        service="stage1_oncology_ml_integration",
        model_loaded=pipeline_ready,
        db_connected=db_ready,
        version="1.0.0"
    )

@app.get("/model/metadata", tags=["Model Governance"])
def get_model_metadata():
    """Returns active model metadata, CV accuracy, and clinical specifications."""
    return {
        "model_architecture": "Calibrated Regularized XGBoost (Platt Scaling)",
        "framework": "xgboost / scikit-learn",
        "training_patients": 4377,
        "5_fold_cv_accuracy": 0.9516,
        "held_out_test_accuracy": 0.9393,
        "generalization_gap": 0.0280,
        "high_risk_sensitivity": 0.9474,
        "critical_false_negatives": 0,
        "classes": ["LOW", "MODERATE", "HIGH"],
        "total_features": 96,
        "status": "Production Calibrated"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single_patient(patient: PatientClinicalInput):
    """
    Real-time inference endpoint for a single clinical encounter.
    Persists encounter to the clinical audit database.
    """
    t0 = time.time()
    pipeline = get_pipeline()
    audit_db = get_audit_db()
    
    patient_dict = patient.model_dump() if hasattr(patient, "model_dump") else patient.dict()
    patient_id = patient_dict.pop("patient_id", "PAT-UNKNOWN")
    encounter_id = patient_dict.pop("encounter_id", "ENC-UNKNOWN")
    
    try:
        result = pipeline.predict_single(patient_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")
        
    latency_ms = round((time.time() - t0) * 1000, 2)
    
    # Audit trail persistence
    try:
        audit_db.log_encounter(
            encounter_id=encounter_id,
            patient_id=patient_id,
            prediction_result=result,
            execution_time_ms=latency_ms,
            source_system="REST_API_SINGLE",
            input_payload=patient_dict
        )
        logged = True
    except Exception:
        logged = False
        
    return PredictionResponse(
        patient_id=patient_id,
        encounter_id=encounter_id,
        predicted_risk_tier=result["predicted_risk_tier"],
        confidence_score=result["confidence_score"],
        class_probabilities=result["class_probabilities"],
        clinical_alert=result["clinical_alert"],
        execution_time_ms=latency_ms,
        audit_logged=logged
    )

@app.post("/predict/batch", tags=["Inference"])
def predict_batch_patients(batch: BatchPredictionRequest):
    """
    Batch processing endpoint for hospital cohorts or clinical trials.
    """
    t0 = time.time()
    pipeline = get_pipeline()
    audit_db = get_audit_db()
    
    raw_list = [p.model_dump() if hasattr(p, "model_dump") else p.dict() for p in batch.patients]
    import pandas as pd
    df = pd.DataFrame(raw_list)
    
    patient_ids = df["patient_id"].tolist() if "patient_id" in df.columns else [f"P_{i}" for i in range(len(df))]
    encounter_ids = df["encounter_id"].tolist() if "encounter_id" in df.columns else [f"E_{i}" for i in range(len(df))]
    
    # Run vectorized batch inference
    results_df = pipeline.predict_batch(df)
    total_time_ms = round((time.time() - t0) * 1000, 2)
    
    response_items = []
    for i, row in results_df.iterrows():
        p_id = str(patient_ids[i])
        e_id = str(encounter_ids[i])
        res_dict = {
            "predicted_risk_tier": row["predicted_risk_tier"],
            "confidence_score": float(row["confidence_score"]),
            "class_probabilities": {
                "LOW": float(row["prob_low"]),
                "MODERATE": float(row["prob_moderate"]),
                "HIGH": float(row["prob_high"])
            },
            "clinical_alert": row["clinical_alert"]
        }
        
        # Log to DB
        audit_db.log_encounter(
            encounter_id=e_id,
            patient_id=p_id,
            prediction_result=res_dict,
            execution_time_ms=total_time_ms / len(batch.patients),
            source_system="REST_API_BATCH"
        )
        
        response_items.append({
            "patient_id": p_id,
            "encounter_id": e_id,
            **res_dict
        })
        
    return {
        "total_patients_processed": len(response_items),
        "total_time_ms": total_time_ms,
        "average_latency_per_patient_ms": round(total_time_ms / max(1, len(response_items)), 2),
        "results": response_items
    }

@app.post("/ehr/ingest", tags=["EHR Interoperability"])
def ingest_ehr_payload(bundle: Dict[str, Any]):
    """
    Direct ingestion endpoint for standard HL7 / FHIR JSON Bundles
    exported from Hospital Information Systems.
    """
    t0 = time.time()
    patient_id, encounter_id, canonical_dict = EHRAdapter.parse_fhir_bundle(bundle)
    
    pipeline = get_pipeline()
    audit_db = get_audit_db()
    
    result = pipeline.predict_single(canonical_dict)
    latency_ms = round((time.time() - t0) * 1000, 2)
    
    audit_db.log_encounter(
        encounter_id=encounter_id,
        patient_id=patient_id,
        prediction_result=result,
        execution_time_ms=latency_ms,
        source_system="FHIR_EHR_ADAPTER",
        input_payload=canonical_dict
    )
    
    return {
        "patient_id": patient_id,
        "encounter_id": encounter_id,
        "adapter_status": "SUCCESS_PARSED",
        "predicted_risk_tier": result["predicted_risk_tier"],
        "confidence_score": result["confidence_score"],
        "class_probabilities": result["class_probabilities"],
        "clinical_alert": result["clinical_alert"],
        "execution_time_ms": latency_ms
    }

@app.get("/audit/encounters", tags=["Audit & Governance"])
def get_recent_encounters(limit: int = Query(default=20, ge=1, le=1000)):
    """Retrieves recent clinical audit records."""
    audit_db = get_audit_db()
    return audit_db.get_all_encounters(limit=limit)

@app.get("/audit/patient/{patient_id}", tags=["Audit & Governance"])
def get_patient_audit_history(patient_id: str):
    """Retrieves the full historical timeline of risk assessments for a specific patient."""
    audit_db = get_audit_db()
    records = audit_db.get_patient_encounters(patient_id)
    return {
        "patient_id": patient_id,
        "total_evaluations": len(records),
        "history": records
    }

@app.get("/audit/stats", tags=["Audit & Governance"])
def get_audit_statistics():
    """Returns clinical volume, risk distributions, and latency metrics."""
    audit_db = get_audit_db()
    return audit_db.get_statistics()

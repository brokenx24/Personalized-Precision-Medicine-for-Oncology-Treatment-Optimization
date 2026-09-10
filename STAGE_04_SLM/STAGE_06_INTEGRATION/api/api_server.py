"""
FastAPI Server for Stage 06 Integration Pipeline.
Provides standardized REST endpoints for multi-modal oncology research integration.
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import Dict, Any

from pipeline.integrated_pipeline import IntegratedPipeline
from api.request_schemas import EncounterRequest
from api.response_schemas import EncounterResponse
from integration_logging.audit_logger import AuditLogger

app = FastAPI(
    title="Oncology Treatment Optimization - Stage 06 Integrated Serving API",
    description="End-to-end multimodal integration of ML, DL, NLP, and SLM for oncology research.",
    version="1.0.0"
)

pipeline = IntegratedPipeline()
audit_logger = AuditLogger()

@app.on_event("startup")
def startup_event():
    pipeline.initialize_all()

@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "service": "STAGE_06_INTEGRATION_API",
        "governance": "NO_AUTONOMOUS_CLINICAL_DECISIONS_ALLOWED"
    }

@app.post("/integrate/encounter", response_model=Dict[str, Any])
def process_encounter(request: EncounterRequest):
    payload = request.dict()
    result = pipeline.run(payload)
    
    # Audit log
    audit_logger.log_encounter(
        request_id=result.get("request_id", "UNKNOWN"),
        patient_id=request.patient_id,
        payload=result
    )
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

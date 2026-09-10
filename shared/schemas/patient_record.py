"""
Unified multimodal patient input schema.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class TabularClinicalFeatures(BaseModel):
    patient_id: str
    age: int
    gender: str
    cancer_type: str
    stage: str
    ecog_performance_status: int = Field(ge=0, le=4)
    laboratory_values: Dict[str, float] = Field(default_factory=dict)
    genomic_variants: List[str] = Field(default_factory=list)

class MultimodalInputs(BaseModel):
    patient_id: str
    clinical: TabularClinicalFeatures
    pathology_wsi_tile_paths: List[str] = Field(default_factory=list)
    radiology_dicom_slice_paths: List[str] = Field(default_factory=list)
    biomarker_time_series: List[Dict[str, float]] = Field(default_factory=list)
    unstructured_clinical_note: str

class IntegratedClinicalAssessment(BaseModel):
    patient_id: str
    cancer_type: str
    ml_treatment_response_probability: float
    dl_multimodal_survival_estimate_months: float
    nlp_extracted_entities: List[Dict[str, Any]]
    nlp_urgency_triage: str
    slm_concise_summary: str
    safety_audit_status: str

"""
Request Schemas for Integration API.
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class EncounterRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    clinical_features: Dict[str, Any] = Field(..., description="Tabular clinical and genomic attributes")
    clinical_notes: Optional[str] = Field(default="", description="Free-text clinical encounter note")
    pathology_image: Optional[str] = Field(default=None, description="Path or base64 string of pathology tile")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contextual encounter metadata")

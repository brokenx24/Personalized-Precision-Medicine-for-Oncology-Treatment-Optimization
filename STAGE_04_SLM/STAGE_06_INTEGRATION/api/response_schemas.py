"""
Response Schemas for Integration API.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class GovernancePolicyBlock(BaseModel):
    autonomous_clinical_decision: str = "FORBIDDEN"
    prescriptions_allowed: bool = False
    regulatory_status: str = "RESEARCH_PROTOTYPE_NOT_FOR_CLINICAL_USE"
    human_in_the_loop_required: bool = True
    disclaimer: str

class SafetyBlock(BaseModel):
    safety_status: str
    hallucination_rate: float
    unsupported_claim_count: int
    passed_clinical_boundary: bool
    approved_for_presentation: bool

class EncounterResponse(BaseModel):
    patient_id: str
    request_id: str
    pipeline_version: str
    timestamp: str
    status: str
    provenance: Dict[str, Any]
    clinical_assessments: Dict[str, Any]
    safety_and_governance: SafetyBlock
    governance_policy: GovernancePolicyBlock
    execution_metrics: Dict[str, Any]

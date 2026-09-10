"""
Stage 1 Integration Module
Provides EHR integration, FastAPI microservices, clinical audit database logging, and client SDKs.
"""

from .ehr_adapter import EHRAdapter
from .db_connector import ClinicalAuditDB
from .api import app

__all__ = ["EHRAdapter", "ClinicalAuditDB", "app"]

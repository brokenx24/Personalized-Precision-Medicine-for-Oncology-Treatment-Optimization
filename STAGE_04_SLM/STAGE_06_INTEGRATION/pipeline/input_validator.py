"""
Input Validator for Stage 06 Integration Pipeline.
Validates multi-modal patient encounter data against strict schemas.
"""

from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger("InputValidator")

class InputValidator:
    def __init__(self):
        self.required_fields = ["patient_id", "clinical_features"]
        self.optional_fields = ["pathology_image", "clinical_notes", "metadata"]
        
    def validate(self, payload: Dict[str, Any]) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
        errors = []
        warnings = []
        
        if not isinstance(payload, dict):
            return False, ["Payload must be a dictionary"], warnings, {}
            
        for req in self.required_fields:
            if req not in payload or payload[req] is None:
                errors.append(f"Missing required field: '{req}'")
                
        if errors:
            return False, errors, warnings, {}
            
        # Validate patient_id
        patient_id = str(payload.get("patient_id", "")).strip()
        if not patient_id:
            errors.append("Field 'patient_id' cannot be empty")
            
        # Validate clinical_features
        features = payload.get("clinical_features")
        if not isinstance(features, dict):
            errors.append("Field 'clinical_features' must be a key-value dictionary")
        else:
            if "age" in features:
                try:
                    age = float(features["age"])
                    if age < 0 or age > 125:
                        warnings.append(f"Unusual age value: {age}")
                except (ValueError, TypeError):
                    errors.append("Feature 'age' must be numeric")
            if "tumor_size_cm" in features:
                try:
                    ts = float(features["tumor_size_cm"])
                    if ts < 0:
                        errors.append("Feature 'tumor_size_cm' cannot be negative")
                except (ValueError, TypeError):
                    errors.append("Feature 'tumor_size_cm' must be numeric")

        # Validate notes
        notes = payload.get("clinical_notes", "")
        if notes and not isinstance(notes, str):
            warnings.append("Field 'clinical_notes' should be a string; converting")
            payload["clinical_notes"] = str(notes)
            
        is_valid = len(errors) == 0
        return is_valid, errors, warnings, payload

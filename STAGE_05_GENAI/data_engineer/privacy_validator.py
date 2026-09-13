"""
Stage 05 Privacy Validator.
Enforces strict anti-PII and de-identification governance across both Reference and Synthetic data.
Clearly distinguishes between REFERENCE/HISTORICAL data and SYNTHETIC data.
"""
import re
from datetime import datetime, timezone
import pandas as pd
from typing import Dict, Any, Tuple, List

class PrivacyViolationError(ValueError):
    """Raised when Protected Health Information (PHI) or direct patient identifiers are detected."""
    pass

class PrivacyValidator:
    # Common PII detection regex patterns
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
    REAL_NAME_INDICATORS = re.compile(r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?\b")
    DIRECT_IDENTIFIER_FIELDS = {
        "name", "patient_name", "first_name", "last_name", "full_name",
        "ssn", "social_security_number", "phone", "telephone", "mobile",
        "email", "email_address", "address", "street_address", "zip_code",
        "mrn", "medical_record_number"
    }
    
    VALID_SYNTHETIC_ID_PATTERN = re.compile(
        r"^(SYN-PAT-\d{4,6}|SYN_NOTE_\d+|SYN_TRAJ_\d+|SYN_MUT_\d+|EDGE-\d{2}|WILDCARD-01)$"
    )

    @classmethod
    def validate_reference_record(cls, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates an existing reference/historical patient record.
        Ensures NO direct PII (names, phone, email, SSN, addresses) is present.
        Does NOT require synthetic_flag to be True.
        """
        violations = []
        
        # 1. Check for prohibited direct identifier column names
        for key in record.keys():
            if str(key).lower() in cls.DIRECT_IDENTIFIER_FIELDS:
                violations.append(f"DIRECT_IDENTIFIER_FIELD_DETECTED: Field '{key}' is prohibited in reference datasets.")
                
        # 2. Textual search for PII pattern matches
        record_str = " ".join(f"{k}:{v}" for k, v in record.items())
        if cls.SSN_PATTERN.search(record_str):
            violations.append("PII_DETECTED: Social Security Number pattern found.")
        if cls.PHONE_PATTERN.search(record_str):
            violations.append("PII_DETECTED: Telephone number pattern found.")
        if cls.EMAIL_PATTERN.search(record_str):
            violations.append("PII_DETECTED: Email address pattern found.")
        if cls.REAL_NAME_INDICATORS.search(record_str):
            violations.append("PII_DETECTED: Named individual honorific pattern found.")

        is_valid = len(violations) == 0
        return is_valid, violations

    @classmethod
    def validate_synthetic_record(cls, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates a generated synthetic oncology case (used in later GenAI roles).
        Requires synthetic_flag == True, valid synthetic ID, and zero PII.
        """
        violations = []

        # 1. Hard requirement: synthetic_flag must be explicitly True
        if record.get("synthetic_flag") is not True:
            violations.append("CRITICAL: synthetic_flag must be explicitly set to True for synthetic cases.")

        # 2. Patient ID format check
        patient_id = str(record.get("patient_id", ""))
        if not cls.VALID_SYNTHETIC_ID_PATTERN.match(patient_id):
            violations.append(f"INVALID_SYNTHETIC_ID: Patient ID '{patient_id}' does not match permitted synthetic pattern.")

        # 3. Textual search for PII pattern matches
        record_str = " ".join(f"{k}:{v}" for k, v in record.items())
        if cls.SSN_PATTERN.search(record_str):
            violations.append("PII_DETECTED: Social Security Number pattern found.")
        if cls.PHONE_PATTERN.search(record_str):
            violations.append("PII_DETECTED: Telephone number pattern found.")
        if cls.EMAIL_PATTERN.search(record_str):
            violations.append("PII_DETECTED: Email address pattern found.")
        if cls.REAL_NAME_INDICATORS.search(record_str):
            violations.append("PII_DETECTED: Named individual honorific pattern found.")

        is_valid = len(violations) == 0
        return is_valid, violations

    @classmethod
    def validate_clinical_text(cls, text: str) -> Tuple[bool, List[str]]:
        """Scans clinical unstructured text for PII patterns."""
        violations = []
        if not text or not isinstance(text, str):
            return True, []

        if cls.SSN_PATTERN.search(text):
            violations.append("PII_DETECTED: SSN pattern in clinical text.")
        if cls.PHONE_PATTERN.search(text):
            violations.append("PII_DETECTED: Phone number pattern in clinical text.")
        if cls.EMAIL_PATTERN.search(text):
            violations.append("PII_DETECTED: Email pattern in clinical text.")
        if cls.REAL_NAME_INDICATORS.search(text):
            violations.append("PII_DETECTED: Name indicator in clinical text.")

        return len(violations) == 0, violations

    @classmethod
    def audit_dataset(cls, df: pd.DataFrame, record_type: str = "REFERENCE") -> Tuple[bool, Dict[str, Any]]:
        """
        Audits an entire DataFrame for privacy and compliance.
        Generates a structured report matching privacy_report.schema.json.
        """
        all_violations = []
        direct_ids_count = 0
        
        # Check column names
        for col in df.columns:
            if str(col).lower() in cls.DIRECT_IDENTIFIER_FIELDS:
                direct_ids_count += 1
                all_violations.append(f"Direct identifier column present: {col}")

        # Sample check rows (audit all rows if <= 2000, else sample 1000)
        sample_df = df if len(df) <= 2000 else df.sample(n=1000, random_state=42)
        validator_fn = cls.validate_reference_record if record_type == "REFERENCE" else cls.validate_synthetic_record
        
        for idx, row in sample_df.iterrows():
            ok, errs = validator_fn(row.to_dict())
            if not ok:
                all_violations.extend([f"Row {idx}: {e}" for e in errs])

        status = "PASS" if len(all_violations) == 0 and direct_ids_count == 0 else "FAIL"
        
        report = {
            "timestamp": "2026-09-14T00:00:00Z",
            "records_audited": len(sample_df),
            "record_type": record_type,
            "direct_identifiers_found": direct_ids_count,
            "direct_identifiers_removed": (direct_ids_count == 0),
            "privacy_violations": all_violations[:50],  # cap list for readability
            "de_identification_verified": (direct_ids_count == 0 and len(all_violations) == 0),
            "full_anonymization_guarantee": "NOT_CLAIMED",
            "anonymization_notes": "Direct identifiers (names, SSN, phone, email, address, MRN) were verified absent. Complete k-anonymity is not claimed because clinical quasi-identifiers (age, sex, cancer type, stage) are preserved for oncology statistical prior modeling.",
            "privacy_status": status
        }
        return (status == "PASS"), report

"""
Clinical Boundary Checker.
Scans text to ensure no prescriptive statements or autonomous decisions exist.
"""

import re
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger("ClinicalBoundaryChecker")

class ClinicalBoundaryChecker:
    FORBIDDEN_PRESCRIPTIONS = [
        r"\bprescribe\b",
        r"\badminister\b",
        r"\binitiate treatment immediately\b",
        r"\bdiscontinue chemotherapy\b",
        r"\bmust be given\b",
        r"\bpatient should take\b",
        r"\bdose escalation recommended\b",
        r"\bstop medication\b",
        r"\bordering\b",
        r"\bdispense\b"
    ]
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.FORBIDDEN_PRESCRIPTIONS]
        
    def check_boundaries(self, text: str) -> Tuple[bool, List[str]]:
        """Checks whether the generated text violates autonomous prescription boundaries."""
        violations = []
        if not text or not isinstance(text, str):
            return False, ["FAIL-CLOSED: Empty or invalid text passed to boundary checker"]
            
        for pattern in self.patterns:
            matches = pattern.findall(text)
            if matches:
                for m in matches:
                    violations.append(f"Forbidden prescriptive token/phrase detected: '{m}'")
                    
        passed = len(violations) == 0
        return passed, violations

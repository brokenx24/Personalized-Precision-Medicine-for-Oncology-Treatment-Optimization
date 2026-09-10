import re
import pandas as pd

class PIISanitizer:
    def __init__(self):
        self.patterns = [
            (r'Report signed by synthetic clinician Dr\.\s*[A-Za-z]+', '[REDACTED_CLINICIAN]'),
            (r'Phone:\s*\d{3}-\d{4}', '[REDACTED_PHONE]'),
            (r'Email:\s*[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]'),
            (r'MRN:\s*\d{6,}', '[REDACTED_MRN]')
        ]
        self.sanitization_audit = []

    def sanitize(self, text, record_id, field_name):
        if not text:
            return ""
        sanitized = str(text)
        for pat, repl in self.patterns:
            matches = re.findall(pat, sanitized)
            if matches:
                for m in matches:
                    self.sanitization_audit.append({
                        "record_id": record_id,
                        "field_name": field_name,
                        "matched_pattern": m,
                        "replacement": repl
                    })
                sanitized = re.sub(pat, repl, sanitized)
        return sanitized

    def get_audit_df(self):
        return pd.DataFrame(self.sanitization_audit)

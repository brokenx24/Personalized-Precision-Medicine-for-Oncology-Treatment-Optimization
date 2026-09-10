import hashlib
import pandas as pd

def compute_hash(text):
    if not text:
        return ""
    return hashlib.sha256(str(text).strip().lower().encode('utf-8')).hexdigest()

class DuplicateHandler:
    def __init__(self):
        self.seen_row_hashes = set()
        self.report_to_summary = {}
        self.audit_log = []

    def check_duplicate(self, record_id, patient_id, report_text, summary_text):
        rep_hash = compute_hash(report_text)
        sum_hash = compute_hash(summary_text)
        pair_hash = hashlib.sha256(f"{rep_hash}:{sum_hash}".encode('utf-8')).hexdigest()

        # Check exact pair duplicate
        if pair_hash in self.seen_row_hashes:
            self.audit_log.append({
                "record_id": record_id,
                "patient_id": patient_id,
                "issue": "EXACT_DUPLICATE",
                "action": "QUARANTINED"
            })
            return True, "EXACT_DUPLICATE"

        # Check conflicting summary for same report
        if rep_hash in self.report_to_summary and self.report_to_summary[rep_hash] != sum_hash:
            self.audit_log.append({
                "record_id": record_id,
                "patient_id": patient_id,
                "issue": "CONFLICTING_SUMMARY_DUPLICATE",
                "action": "QUARANTINED"
            })
            return True, "CONFLICTING_SUMMARY_DUPLICATE"

        self.seen_row_hashes.add(pair_hash)
        self.report_to_summary[rep_hash] = sum_hash
        return False, "UNIQUE"

    def get_audit_df(self):
        return pd.DataFrame(self.audit_log)

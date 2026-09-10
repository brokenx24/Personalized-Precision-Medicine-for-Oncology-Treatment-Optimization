import re
import json
import pandas as pd

class QualityFilter:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.pat_regex = re.compile(self.cfg['patient_id_pattern'])
        self.rec_regex = re.compile(self.cfg['record_id_pattern'])
        self.rejected_records = []

    def count_sentences(self, text):
        if not text or str(text).strip() == "":
            return 0
        sents = [s.strip() for s in re.split(r'[.!?]+', str(text)) if s.strip()]
        return len(sents)

    def validate_record(self, row):
        rec_id = row.get('record_id', '')
        pat_id = row.get('patient_id', '')
        rep = str(row.get('clinical_report', '') or '').strip()
        summ = str(row.get('target_summary', '') or '').strip()

        # 1. Missing / Empty texts
        if not rep:
            self.rejected_records.append({
                "record_id": rec_id,
                "patient_id": pat_id,
                "rejection_reason": "EMPTY_REPORT",
                "details": "Clinical report is null, empty or whitespace"
            })
            return False, "EMPTY_REPORT"

        if not summ:
            self.rejected_records.append({
                "record_id": rec_id,
                "patient_id": pat_id,
                "rejection_reason": "EMPTY_SUMMARY",
                "details": "Target summary is null, empty or whitespace"
            })
            return False, "EMPTY_SUMMARY"

        # 2. ID patterns
        if not self.pat_regex.match(str(pat_id)):
            self.rejected_records.append({
                "record_id": rec_id,
                "patient_id": pat_id,
                "rejection_reason": "INVALID_PATIENT_ID",
                "details": f"Patient ID '{pat_id}' does not match pattern"
            })
            return False, "INVALID_PATIENT_ID"

        if not self.rec_regex.match(str(rec_id)):
            self.rejected_records.append({
                "record_id": rec_id,
                "patient_id": pat_id,
                "rejection_reason": "INVALID_RECORD_ID",
                "details": f"Record ID '{rec_id}' does not match pattern"
            })
            return False, "INVALID_RECORD_ID"

        # 3. Length bounds
        if len(rep) < self.cfg['min_report_characters']:
            self.rejected_records.append({
                "record_id": rec_id,
                "patient_id": pat_id,
                "rejection_reason": "REPORT_TOO_SHORT",
                "details": f"Report length {len(rep)} < {self.cfg['min_report_characters']}"
            })
            return False, "REPORT_TOO_SHORT"

        if len(rep) > self.cfg['max_report_characters']:
            self.rejected_records.append({
                "record_id": rec_id,
                "patient_id": pat_id,
                "rejection_reason": "REPORT_TOO_LONG",
                "details": f"Report length {len(rep)} > {self.cfg['max_report_characters']}"
            })
            return False, "REPORT_TOO_LONG"

        if len(summ) < self.cfg['min_summary_characters']:
            self.rejected_records.append({
                "record_id": rec_id,
                "patient_id": pat_id,
                "rejection_reason": "SUMMARY_TOO_SHORT",
                "details": f"Summary length {len(summ)} < {self.cfg['min_summary_characters']}"
            })
            return False, "SUMMARY_TOO_SHORT"

        if len(summ) > self.cfg['max_summary_characters']:
            self.rejected_records.append({
                "record_id": rec_id,
                "patient_id": pat_id,
                "rejection_reason": "SUMMARY_TOO_LONG",
                "details": f"Summary length {len(summ)} > {self.cfg['max_summary_characters']}"
            })
            return False, "SUMMARY_TOO_LONG"

        # 4. Sentence count bounds
        s_count = self.count_sentences(summ)
        if s_count < self.cfg['min_summary_sentences'] or s_count > self.cfg['max_summary_sentences']:
            self.rejected_records.append({
                "record_id": rec_id,
                "patient_id": pat_id,
                "rejection_reason": "SENTENCE_COUNT_OUT_OF_BOUNDS",
                "details": f"Sentence count {s_count} not in [{self.cfg['min_summary_sentences']}, {self.cfg['max_summary_sentences']}]"
            })
            return False, "SENTENCE_COUNT_OUT_OF_BOUNDS"

        # 5. Length ratio
        ratio = len(rep) / max(len(summ), 1)
        if ratio < self.cfg['min_report_summary_ratio'] or ratio > self.cfg['max_report_summary_ratio']:
            self.rejected_records.append({
                "record_id": rec_id,
                "patient_id": pat_id,
                "rejection_reason": "EXTREME_LENGTH_RATIO",
                "details": f"Length ratio {ratio:.2f} out of bounds"
            })
            return False, "EXTREME_LENGTH_RATIO"

        return True, "VALID"

    def get_rejected_df(self):
        return pd.DataFrame(self.rejected_records)

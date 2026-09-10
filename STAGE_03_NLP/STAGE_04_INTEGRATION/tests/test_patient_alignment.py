import unittest
import pandas as pd
from STAGE_04_INTEGRATION.alignment.patient_alignment import PatientAlignmentEngine

class TestPatientAlignment(unittest.TestCase):
    def test_alignment_logic(self):
        eng = PatientAlignmentEngine()
        df1 = pd.DataFrame({"patient_id": ["P1", "P2"]})
        df2 = pd.DataFrame({"patient_id": ["P2", "P3"]})
        df3 = pd.DataFrame({"patient_id": ["P3", "P4"]})
        res = eng.align_cohorts(df1, df2, df3)
        self.assertIn("alignment_df", res)
        self.assertIn("statistics", res)

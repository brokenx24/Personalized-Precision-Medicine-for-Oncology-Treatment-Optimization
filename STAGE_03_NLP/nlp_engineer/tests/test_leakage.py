"""Unit Tests for Zero Cross-Split Patient Leakage Invariant.
NLP Engineer Module - Stage 03 NLP.
"""

import unittest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath("."))

class TestPatientLeakage(unittest.TestCase):
    def test_zero_leakage_invariant(self):
        df_tr = pd.read_csv("STAGE_03_NLP/data_engineer/splits/train.csv")
        df_va = pd.read_csv("STAGE_03_NLP/data_engineer/splits/validation.csv")
        df_te = pd.read_csv("STAGE_03_NLP/data_engineer/splits/test.csv")
        
        p_tr = set(df_tr["patient_id"].unique())
        p_va = set(df_va["patient_id"].unique())
        p_te = set(df_te["patient_id"].unique())
        
        self.assertEqual(len(p_tr & p_va), 0, "Train-Val patient overlap must be 0")
        self.assertEqual(len(p_tr & p_te), 0, "Train-Test patient overlap must be 0")
        self.assertEqual(len(p_va & p_te), 0, "Val-Test patient overlap must be 0")

if __name__ == "__main__":
    unittest.main()

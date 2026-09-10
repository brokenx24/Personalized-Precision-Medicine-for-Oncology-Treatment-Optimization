"""
Unit Tests for Stage 04 SLM Data Engineer Subsystem
"""

import os
import json
import unittest
import pandas as pd

class TestStage04SLMDataEngineer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = "c:/Users/shyam/OneDrive/Documents/HOSPITAL"
        cls.base_dir = os.path.join(cls.project_root, "STAGE_04_SLM")
        cls.raw_csv = os.path.join(cls.base_dir, "data_engineer", "raw", "raw_oncology_summarization.csv")
        cls.clean_csv = os.path.join(cls.base_dir, "data_engineer", "cleaned", "cleaned_oncology_summarization.csv")
        cls.split_audit_json = os.path.join(cls.base_dir, "data_engineer", "splits", "patient_split_audit.json")
        cls.train_jsonl = os.path.join(cls.base_dir, "data_engineer", "splits", "train.jsonl")
        cls.val_jsonl = os.path.join(cls.base_dir, "data_engineer", "splits", "validation.jsonl")
        cls.test_jsonl = os.path.join(cls.base_dir, "data_engineer", "splits", "test.jsonl")

    def test_01_raw_dataset_exists_and_populated(self):
        self.assertTrue(os.path.exists(self.raw_csv), "Raw dataset CSV does not exist")
        df = pd.read_csv(self.raw_csv, nrows=10)
        self.assertGreater(len(df), 0, "Raw dataset CSV is empty")
        expected_cols = ["patient_id", "record_id", "clinical_report", "target_summary", "data_quality_status"]
        for c in expected_cols:
            self.assertIn(c, df.columns, f"Column {c} missing from raw dataset")

    def test_02_cleaned_dataset_validity(self):
        self.assertTrue(os.path.exists(self.clean_csv), "Cleaned dataset CSV does not exist")
        df = pd.read_csv(self.clean_csv)
        self.assertGreater(len(df), 20000, "Cleaned dataset contains fewer records than expected")
        self.assertEqual(df['clinical_report'].isna().sum(), 0, "Null clinical_report found in cleaned dataset")
        self.assertEqual(df['target_summary'].isna().sum(), 0, "Null target_summary found in cleaned dataset")
        self.assertEqual((df['clinical_report'].str.strip() == '').sum(), 0, "Empty string clinical_report found")
        self.assertEqual((df['target_summary'].str.strip() == '').sum(), 0, "Empty string target_summary found")

    def test_03_zero_patient_leakage(self):
        self.assertTrue(os.path.exists(self.split_audit_json), "Patient split audit JSON missing")
        with open(self.split_audit_json, "r", encoding="utf-8") as f:
            audit = json.load(f)
        leakage = audit["patient_leakage_verification"]
        self.assertEqual(leakage["train_val_intersection"], 0, "Train-Val patient leakage detected")
        self.assertEqual(leakage["train_test_intersection"], 0, "Train-Test patient leakage detected")
        self.assertEqual(leakage["val_test_intersection"], 0, "Val-Test patient leakage detected")
        self.assertEqual(leakage["status"], "PASS")

    def test_04_slm_instruction_format(self):
        for path in [self.train_jsonl, self.val_jsonl, self.test_jsonl]:
            self.assertTrue(os.path.exists(path), f"JSONL split file missing: {path}")
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                data = json.loads(first_line)
                self.assertIn("instruction", data)
                self.assertIn("input", data)
                self.assertIn("output", data)
                self.assertIn("metadata", data)
                self.assertTrue(len(data["input"]) > 50, "Input report text too short")
                self.assertTrue(len(data["output"]) > 30, "Output summary text too short")

    def test_05_reproducibility_artifacts_exist(self):
        repro_path = os.path.join(self.base_dir, "data_engineer", "outputs", "reproducibility_report.json")
        self.assertTrue(os.path.exists(repro_path), "Reproducibility report JSON missing")
        with open(repro_path, "r", encoding="utf-8") as f:
            repro = json.load(f)
        self.assertEqual(repro["random_seed"], 42)
        self.assertIn("raw_oncology_summarization.csv", repro["artifact_hashes"])

if __name__ == "__main__":
    unittest.main()

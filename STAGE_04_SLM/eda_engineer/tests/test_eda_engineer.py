import os
import json
import unittest
import pandas as pd

class TestStage04SLMEDAEngineer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = "c:/Users/shyam/OneDrive/Documents/HOSPITAL"
        cls.eda_dir = os.path.join(cls.project_root, "STAGE_04_SLM", "eda_engineer")
        cls.out_dir = os.path.join(cls.eda_dir, "outputs")
        cls.clean_csv = os.path.join(cls.project_root, "STAGE_04_SLM", "data_engineer", "cleaned", "cleaned_oncology_summarization.csv")

    def test_01_dataset_loading_and_shape(self):
        self.assertTrue(os.path.exists(self.clean_csv), "Cleaned dataset CSV missing")
        df = pd.read_csv(self.clean_csv)
        self.assertEqual(len(df), 23353, f"Expected 23,353 records, found {len(df)}")
        self.assertIn("clinical_report", df.columns)
        self.assertIn("target_summary", df.columns)

    def test_02_token_statistics_percentiles(self):
        path = os.path.join(self.out_dir, "token_statistics.json")
        self.assertTrue(os.path.exists(path), "token_statistics.json missing")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("complete_sequence_tokens", data)
        self.assertLess(data["complete_sequence_tokens"]["p99"], 500, "P99 complete sequence length too high")

    def test_03_medical_token_retention(self):
        mut_path = os.path.join(self.out_dir, "mutation_analysis.json")
        dose_path = os.path.join(self.out_dir, "dosage_analysis.json")
        with open(mut_path, "r", encoding="utf-8") as f:
            mut_data = json.load(f)
        with open(dose_path, "r", encoding="utf-8") as f:
            dose_data = json.load(f)
        self.assertGreaterEqual(mut_data["mutation_retention_percentage"], 90.0)
        self.assertGreaterEqual(dose_data["dosage_retention_percentage"], 20.0, "Dosage retention unexpectedly low")

    def test_04_patient_leakage_zero(self):
        path = os.path.join(self.out_dir, "leakage_analysis.json")
        self.assertTrue(os.path.exists(path), "leakage_analysis.json missing")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        leak = data["patient_leakage"]
        self.assertEqual(leak["train_validation_overlap"], 0)
        self.assertEqual(leak["train_test_overlap"], 0)
        self.assertEqual(leak["validation_test_overlap"], 0)
        self.assertEqual(leak["status"], "PASS")

    def test_05_truncation_risk_zero_at_512(self):
        path = os.path.join(self.out_dir, "truncation_risk_report.json")
        self.assertTrue(os.path.exists(path), "truncation_risk_report.json missing")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["context_512"]["exceedance_percentage"], 0.0)
        self.assertEqual(data["context_512"]["truncation_risk_level"], "ZERO")

if __name__ == "__main__":
    unittest.main()

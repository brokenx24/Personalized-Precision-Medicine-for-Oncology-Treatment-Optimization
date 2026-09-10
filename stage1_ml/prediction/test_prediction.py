"""
================================================================================
UNIT TEST SUITE FOR ONCOLOGY PREDICTION PIPELINE
================================================================================
Executes unit tests verifying that `OncologyPredictionPipeline` produces valid,
medically consistent, and properly formatted risk predictions across diverse
patient profiles (Ovarian, Lung, Breast, and Colon cohorts).
================================================================================
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stage1_ml.prediction.prediction import OncologyPredictionPipeline

class TestOncologyPredictionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = OncologyPredictionPipeline(project_root=project_root)
        
    def test_single_high_risk_patient(self):
        """Test metastatic Stage IV patient returns HIGH risk tier"""
        payload = {
            'patient_id': 'TEST-MET-001',
            'cancer_type': 'Lung Adenocarcinoma',
            'cancer_stage': 'Stage IV',
            'tumor_grade': 'G3/G4',
            'path_t_stage': 'T3',
            'path_n_stage': 'N2',
            'path_m_stage': 'M1',
            'ctdna_baseline_maf': 0.52,
            'mutation_count': 320,
            'platelets_10_3_ul': 340,
            'hemoglobin_g_dl': 10.5
        }
        res = self.pipeline.predict(payload)
        self.assertEqual(res['predicted_risk_tier'], 'HIGH')
        self.assertEqual(res['alert_level'], 'red')
        self.assertIn('HIGH', res['risk_probabilities'])
        self.assertGreater(res['confidence'], 0.70)
        
    def test_single_low_risk_patient(self):
        """Test localized early Stage I patient returns LOW risk tier"""
        payload = {
            'patient_id': 'TEST-EARLY-001',
            'cancer_type': 'Breast Invasive Carcinoma',
            'cancer_stage': 'Stage I',
            'tumor_grade': 'G1',
            'path_t_stage': 'T1',
            'path_n_stage': 'N0',
            'path_m_stage': 'M0',
            'ctdna_baseline_maf': 0.005,
            'mutation_count': 12,
            'platelets_10_3_ul': 210,
            'hemoglobin_g_dl': 14.2
        }
        res = self.pipeline.predict(payload)
        self.assertEqual(res['predicted_risk_tier'], 'LOW')
        self.assertEqual(res['alert_level'], 'green')
        self.assertIn('routine surveillance', res['clinical_recommendation'].lower())
        
    def test_probability_distribution_sum(self):
        """Verify class probabilities sum to 1.0 within numerical tolerance"""
        payload = {
            'patient_id': 'TEST-SUM-001',
            'cancer_type': 'Ovarian Serous Cystadenocarcinoma',
            'cancer_stage': 'Stage II',
            'tumor_grade': 'G2',
            'path_t_stage': 'T2',
            'path_n_stage': 'N0',
            'path_m_stage': 'M0'
        }
        res = self.pipeline.predict(payload)
        prob_sum = sum(res['risk_probabilities'].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)
        
    def test_batch_dataframe_prediction(self):
        """Verify batch prediction accepts multi-patient DataFrame"""
        patients = [
            {'patient_id': 'BATCH-001', 'cancer_stage': 'Stage I', 'path_m_stage': 'M0'},
            {'patient_id': 'BATCH-002', 'cancer_stage': 'Stage IV', 'path_m_stage': 'M1'}
        ]
        df = pd.DataFrame(patients)
        results = self.pipeline.predict(df)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['patient_id'], 'BATCH-001')
        self.assertEqual(results[1]['patient_id'], 'BATCH-002')

if __name__ == '__main__':
    unittest.main(verbosity=2)

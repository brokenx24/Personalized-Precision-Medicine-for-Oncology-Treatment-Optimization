"""
================================================================================
INFERENCE & PREDICTION API PIPELINE MODULE
================================================================================
Defines the production `OncologyPredictionPipeline` class for real-time inference,
clinical risk tier classification, and FastAPI integration.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

class OncologyPredictionPipeline:
    """
    Production-grade oncology risk inference pipeline.
    Loads serialized feature preprocessors and trained calibrated classifiers.
    Accepts raw patient dictionary or pandas DataFrame, transforms features,
    and returns risk classification, probabilities, and clinical recommendations.
    """
    def __init__(self, model_path=None, preprocessor_path=None, project_root=None):
        if project_root is None:
            self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.project_root = project_root
            
        if model_path is None:
            self.model_path = os.path.join(self.project_root, "stage1_ml", "models", "best_ml_model.joblib")
        else:
            self.model_path = model_path
            
        if preprocessor_path is None:
            self.preprocessor_path = os.path.join(self.project_root, "stage1_ml", "models", "feature_preprocessor.joblib")
        else:
            self.preprocessor_path = preprocessor_path
            
        self.model = None
        self.scaler = None
        self.encoder = None
        self.num_cols = None
        self.cat_cols = None
        self.classes = ['LOW', 'MODERATE', 'HIGH']
        self._load_artifacts()
        
    def _load_artifacts(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor artifact not found at: {self.preprocessor_path}")
            
        self.model = joblib.load(self.model_path)
        preproc = joblib.load(self.preprocessor_path)
        self.preprocessor = preproc
        
        if isinstance(preproc, dict):
            self.scaler = preproc['scaler']
            self.encoder = preproc['encoder']
            self.num_cols = preproc['num_cols']
            self.cat_cols = preproc['cat_cols']
        else:
            self.scaler = preproc.scaler
            self.encoder = preproc.encoder
            self.num_cols = preproc.num_cols
            self.cat_cols = preproc.cat_cols
            
    def _prepare_dataframe(self, patient_data):
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        elif isinstance(patient_data, pd.DataFrame):
            df = patient_data.copy()
        elif isinstance(patient_data, list):
            df = pd.DataFrame(patient_data)
        else:
            raise ValueError("Input patient_data must be a dict, list of dicts, or a pandas DataFrame")
            
        # Clinical baseline defaults
        defaults = {
            'age': 60.0, 'sex': 'Female', 'weight_kg': 65.0, 'height_cm': 165.0,
            'bmi': 24.0, 'cancer_type': 'Breast Invasive Carcinoma', 'cancer_stage': 'Stage II',
            'tumor_grade': 'G2', 'path_t_stage': 'T2', 'path_n_stage': 'N0', 'path_m_stage': 'M0',
            'performance_status_ecog': 0, 'comorbidity_count': 1, 'hemoglobin_g_dl': 13.5,
            'wbc_10_3_ul': 6.5, 'platelets_10_3_ul': 240.0, 'creatinine_mg_dl': 0.9,
            'bilirubin_mg_dl': 0.5, 'alt_u_l': 25.0, 'ast_u_l': 25.0, 'albumin_g_dl': 4.2,
            'ctdna_baseline_maf': 0.1, 'protein_biomarker_cea_ng_ml': 3.0, 'mutation_count': 50.0,
            'fraction_genome_altered': 0.3, 'aneuploidy_score': 10.0, 'tmb_nonsynonymous': 5.0,
            'msi_sensor_score': 0.5, 'buffa_hypoxia_score': 10.0, 'ragnum_hypoxia_score': 5.0,
            'winter_hypoxia_score': 15.0, 'treatment_radiation': 0, 'treatment_neoadjuvant': 0,
            'days_since_diagnosis': 0
        }
        for k, v in defaults.items():
            if k not in df.columns:
                df[k] = v
            else:
                df[k] = df[k].fillna(v)
                
        # Compute clinical interaction features
        df['ast_alt_ratio'] = df['ast_u_l'] / (df['alt_u_l'] + 1e-5)
        df['alb_creat_ratio'] = df['albumin_g_dl'] / (df['creatinine_mg_dl'] + 1e-5)
        df['systemic_immune_index'] = (df['platelets_10_3_ul'] * df['wbc_10_3_ul']) / (df['hemoglobin_g_dl'] + 1e-5)
        df['biomarker_hypoxia_burden'] = df['ctdna_baseline_maf'] * (df['buffa_hypoxia_score'].fillna(0) + 1.0)
        df['genomic_instability_index'] = df['fraction_genome_altered'].fillna(0) * (df['aneuploidy_score'].fillna(0) + 1.0)
        return df
        
    def predict(self, patient_data):
        """
        Executes prediction on raw input patient payload.
        Returns a single dict if input was single dict, or list of dicts if batch.
        """
        is_single_dict = isinstance(patient_data, dict)
        df = self._prepare_dataframe(patient_data)
        
        # Transform numeric and categorical
        s_num = self.scaler.transform(df[self.num_cols])
        e_cat = self.encoder.transform(df[self.cat_cols])
        X_trans = np.hstack([s_num, e_cat])
        
        preds = self.model.predict(X_trans)
        probs = self.model.predict_proba(X_trans)
        
        results = []
        for i in range(len(df)):
            pred_idx = int(preds[i])
            pred_class = self.classes[pred_idx]
            prob_dist = {self.classes[j]: round(float(probs[i, j]), 4) for j in range(3)}
            conf = round(float(probs[i, pred_idx]), 4)
            
            # Clinical recommendation logic
            if pred_class == 'HIGH':
                action = "Urgent multidisciplinary oncology board referral; evaluate targeted combination therapy."
                risk_color = "red"
            elif pred_class == 'MODERATE':
                action = "Standard adjuvant treatment with bi-monthly ctDNA and lab monitoring."
                risk_color = "amber"
            else:
                action = "Routine surveillance protocol; low risk of early systemic recurrence."
                risk_color = "green"
                
            p_id = df.get('patient_id', pd.Series([f'PATIENT_{i+1:03d}'])).iloc[i]
            results.append({
                'patient_id': str(p_id),
                'predicted_risk_tier': pred_class,
                'confidence': conf,
                'confidence_score': conf,
                'risk_probabilities': prob_dist,
                'class_probabilities': prob_dist,
                'alert_level': risk_color,
                'clinical_alert': risk_color,
                'clinical_recommendation': action,
                'prob_low': prob_dist.get('LOW', 0.0),
                'prob_moderate': prob_dist.get('MODERATE', 0.0),
                'prob_high': prob_dist.get('HIGH', 0.0)
            })
            
        return results[0] if is_single_dict else results

    def predict_single(self, patient_dict: dict) -> dict:
        """Helper for single-patient dictionary inference."""
        res = self.predict(patient_dict)
        return res if isinstance(res, dict) else res[0]

    def predict_batch(self, patient_data):
        """Helper for batch patient inference returning DataFrame or list."""
        results = self.predict(patient_data)
        if isinstance(results, dict):
            results = [results]
        return pd.DataFrame(results)
if __name__ == "__main__":
    pipeline = OncologyPredictionPipeline()
    sample_patient = {
        'patient_id': 'DEMO-PATIENT-001',
        'cancer_type': 'Lung Adenocarcinoma',
        'cancer_stage': 'Stage IV',
        'tumor_grade': 'G3/G4',
        'path_t_stage': 'T3',
        'path_n_stage': 'N2',
        'path_m_stage': 'M1',
        'age': 68,
        'sex': 'Male',
        'ctdna_baseline_maf': 0.45,
        'mutation_count': 250,
        'platelets_10_3_ul': 310,
        'hemoglobin_g_dl': 11.2,
        'wbc_10_3_ul': 8.5
    }
    result = pipeline.predict(sample_patient)
    print("Sample Patient Prediction Result:")
    import json
    print(json.dumps(result, indent=2))

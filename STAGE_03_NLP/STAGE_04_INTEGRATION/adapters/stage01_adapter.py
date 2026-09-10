"""Stage 01 Machine Learning Adapter.
STAGE 04 INTEGRATION SUBSYSTEM.
Locates, validates, and standardizes Stage 01 ML predictions and probabilities.
Strictly Read-Only on STAGE_01_ML.
"""

import os
import json
import numpy as np
import pandas as pd

class Stage01Adapter:
    def __init__(self, config_path="STAGE_04_INTEGRATION/config/paths_config.json"):
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "paths_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.paths = json.load(f)["stage01"]
        self.class_labels = ["LOW", "MODERATE", "HIGH"]
        self._predictions_cache = None

    def validate_artifacts(self) -> dict:
        results = {}
        for k, p in self.paths.items():
            results[k] = os.path.exists(p)
        return results

    def load_predictions(self) -> pd.DataFrame:
        if self._predictions_cache is not None:
            return self._predictions_cache

        df_test = pd.read_csv(self.paths["test_split_path"])
        pids = df_test["patient_id"].tolist()

        eval_data = np.load(self.paths["eval_results_path"], allow_pickle=True)
        y_test_pred = eval_data["y_test_pred"]
        y_test_prob = eval_data["y_test_prob"]

        records = []
        for i, pid in enumerate(pids):
            pred_idx = int(y_test_pred[i])
            pred_cls = self.class_labels[pred_idx]
            probs = y_test_prob[i]
            conf = float(np.max(probs))
            records.append({
                "patient_id": pid,
                "ml_prediction": pred_cls,
                "ml_risk_class": pred_cls,
                "ml_risk_probability": round(float(probs[2]), 4),  # P(HIGH)
                "ml_probability": round(float(probs[pred_idx]), 4),
                "ml_prob_low": round(float(probs[0]), 4),
                "ml_prob_moderate": round(float(probs[1]), 4),
                "ml_prob_high": round(float(probs[2]), 4),
                "ml_confidence": round(conf, 4),
                "ml_ground_truth": df_test.iloc[i].get("oncology_risk_class", None)
            })

        self._predictions_cache = pd.DataFrame(records)
        return self._predictions_cache

    def get_patient_prediction(self, patient_id: str) -> dict:
        df = self.load_predictions()
        match = df[df["patient_id"] == patient_id]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

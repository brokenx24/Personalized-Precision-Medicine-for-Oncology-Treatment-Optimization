"""Multimodal Fusion Engine.
STAGE 04 INTEGRATION SUBSYSTEM.
Executes dynamic, available-modality renormalized fusion:
S_integrated = sum_{m in M_avail} (w_m * P_m(HIGH)) / sum_{m in M_avail} w_m.
Missing modalities are strictly excluded from the denominator rather than penalizing patient risk.
Strictly Read-Only on Upstream Stages.
"""

import json
import os
import numpy as np

class MultimodalFusionEngine:
    def __init__(self, config_path="STAGE_04_INTEGRATION/config/fusion_config.json"):
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "fusion_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.weights = self.config["weights"]
        self.thresholds = self.config["thresholds"]
        self.penalties = self.config.get("missing_modality_penalties", {
            "FULL_MULTIMODAL": 1.0,
            "PARTIAL_MULTIMODAL": 0.85,
            "SINGLE_MODALITY": 0.65
        })

    def fuse_patient(self, ml_record: dict, dl_record: dict, nlp_record: dict) -> dict:
        """Computes integrated oncology risk score across observed modalities."""
        active_weights = {}
        observed_p_high = {}
        confidences = []

        # 1. Stage 01 ML Evidence
        if ml_record is not None and "ml_prob_high" in ml_record:
            active_weights["ML"] = self.weights["ml_weight"]
            observed_p_high["ML"] = float(ml_record["ml_prob_high"])
            confidences.append(float(ml_record.get("ml_confidence", 0.5)))

        # 2. Stage 02 DL Evidence
        if dl_record is not None and "dl_prob_high" in dl_record:
            active_weights["DL"] = self.weights["dl_weight"]
            observed_p_high["DL"] = float(dl_record["dl_prob_high"])
            confidences.append(float(dl_record.get("dl_confidence", 0.5)))

        # 3. Stage 03 NLP Evidence
        if nlp_record is not None and "nlp_high_probability" in nlp_record:
            active_weights["NLP"] = self.weights["nlp_weight"]
            observed_p_high["NLP"] = float(nlp_record["nlp_high_probability"])
            confidences.append(float(nlp_record.get("nlp_confidence", 0.5)))

        # Determine evidence modality status
        observed_modalities = list(active_weights.keys())
        n_obs = len(observed_modalities)

        if n_obs == 3:
            evidence_status = "FULL_MULTIMODAL"
        elif n_obs == 2:
            evidence_status = "PARTIAL_MULTIMODAL"
        elif n_obs == 1:
            evidence_status = "SINGLE_MODALITY"
        else:
            evidence_status = "INSUFFICIENT_DATA"

        missing_modalities = [m for m in ["ML", "DL", "NLP"] if m not in observed_modalities]

        if n_obs == 0:
            return {
                "integrated_risk_score": 0.0,
                "integrated_risk_class": "UNKNOWN",
                "integration_confidence": 0.0,
                "evidence_status": evidence_status,
                "observed_modalities": [],
                "missing_modalities": ["ML", "DL", "NLP"],
                "active_weight_sum": 0.0,
                "modality_contributions": {}
            }

        # Available-modality renormalized weighted fusion:
        # S_integrated = sum_{m in M_avail} (w_m * P_m(HIGH)) / sum_{m in M_avail} w_m
        weight_sum = sum(active_weights.values())
        weighted_sum = sum(active_weights[m] * observed_p_high[m] for m in observed_modalities)
        integrated_score = round(float(weighted_sum / weight_sum), 4)

        # Class determination based on configured thresholds
        if integrated_score < self.thresholds["low_max"]:
            risk_class = "LOW"
        elif integrated_score < self.thresholds["moderate_max"]:
            risk_class = "MODERATE"
        else:
            risk_class = "HIGH"

        # Modality contribution breakdown
        modality_contributions = {}
        for m in observed_modalities:
            eff_weight = active_weights[m] / weight_sum
            modality_contributions[m] = round(float(eff_weight * observed_p_high[m]), 4)

        # Integration confidence scaled by evidence completeness
        base_conf = float(np.mean(confidences)) if confidences else 0.5
        penalty = self.penalties.get(evidence_status, 0.70)
        final_conf = round(float(base_conf * penalty), 4)

        return {
            "integrated_risk_score": integrated_score,
            "integrated_risk_class": risk_class,
            "integration_confidence": final_conf,
            "evidence_status": evidence_status,
            "observed_modalities": observed_modalities,
            "missing_modalities": missing_modalities,
            "active_weight_sum": round(float(weight_sum), 4),
            "modality_contributions": modality_contributions
        }

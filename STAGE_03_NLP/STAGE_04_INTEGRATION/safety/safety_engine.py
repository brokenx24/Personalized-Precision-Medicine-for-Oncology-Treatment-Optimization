"""Clinical Safety Engine.
STAGE 04 INTEGRATION SUBSYSTEM.
Applies transparent clinical safety guardrails, sentinel term detection,
and acute clinical urgency escalation rules.
Strictly Read-Only on Upstream Stages.
"""

import json
import os

class ClinicalSafetyEngine:
    def __init__(self, config_path="STAGE_04_INTEGRATION/config/safety_config.json"):
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "safety_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.sentinel_terms = self.config["sentinel_oncology_terms"]
        self.nlp_hr_thresh = self.config["nlp_high_risk_threshold"]

    def evaluate_safety(self, fused_result: dict, ml_record: dict, dl_record: dict, nlp_record: dict) -> dict:
        """Evaluates 4 transparent clinical safety rules."""
        safety_flag = False
        reasons = []
        rule_triggers = []

        orig_class = fused_result["integrated_risk_class"]
        adjusted_class = orig_class

        # RULE 1: Acute NLP Urgency Override
        # If NLP indicates HIGH urgency with probability >= threshold, escalate to at least MODERATE/HIGH
        if nlp_record and nlp_record.get("nlp_high_probability", 0.0) >= self.nlp_hr_thresh:
            if orig_class == "LOW":
                adjusted_class = "MODERATE"
                safety_flag = True
                rule_triggers.append("RULE_1_ACUTE_URGENCY_ESCALATION")
                reasons.append(f"Acute clinical text probability P(HIGH)={nlp_record['nlp_high_probability']} >= {self.nlp_hr_thresh}; escalated LOW to MODERATE.")

        # RULE 2: Sentinel Oncology Terms & Adverse Events
        detected_sentinels = []
        if nlp_record:
            adverse_events = [ae.lower() for ae in nlp_record.get("adverse_events", [])]
            for term in self.sentinel_terms:
                for ae in adverse_events:
                    if term in ae:
                        detected_sentinels.append(term)
            if detected_sentinels:
                safety_flag = True
                rule_triggers.append("RULE_2_SENTINEL_TERMS_DETECTED")
                reasons.append(f"Critical sentinel oncology toxicity terms detected: {sorted(list(set(detected_sentinels)))}.")

        # RULE 3: Severe Cross-Modal Discordance (NLP HIGH vs Baseline/Imaging LOW)
        nlp_is_high = nlp_record and (nlp_record.get("nlp_urgency_class") == "HIGH")
        ml_is_low = ml_record and (ml_record.get("ml_prediction") == "LOW")
        dl_is_low = dl_record and (dl_record.get("dl_prediction") == "LOW")

        if nlp_is_high and (ml_is_low or dl_is_low):
            safety_flag = True
            rule_triggers.append("RULE_3_SEVERE_DISCORDANCE_WARNING")
            reasons.append("Severe cross-modal discordance: acute clinical note signals HIGH urgency while baseline ML or DL imaging indicates LOW risk.")

        # RULE 4: Missing Modality Alert on Elevated Risk
        is_elevated = (adjusted_class in ["MODERATE", "HIGH"]) or (fused_result["integrated_risk_score"] >= 0.33)
        has_missing = len(fused_result["missing_modalities"]) > 0
        if is_elevated and has_missing:
            safety_flag = True
            rule_triggers.append("RULE_4_MISSING_MODALITY_AT_ELEVATED_RISK")
            reasons.append(f"Elevated risk patient with incomplete modality coverage (Missing: {fused_result['missing_modalities']}). Clinician review recommended.")

        return {
            "safety_flag": safety_flag,
            "original_risk_class": orig_class,
            "final_risk_class": adjusted_class,
            "safety_reasons": reasons,
            "safety_reason_summary": "; ".join(reasons) if reasons else "No active safety alerts.",
            "rules_triggered": rule_triggers,
            "sentinel_terms_found": sorted(list(set(detected_sentinels))),
            "disclaimer": "Synthetic research integration only. This system is not clinically validated."
        }

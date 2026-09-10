"""Stage 03 NLP Adapter.
STAGE 04 INTEGRATION SUBSYSTEM.
Locates, aggregates, and standardizes Stage 03 NLP predictions, probabilities, and NER entities.
Strictly Read-Only on STAGE_03_NLP.
"""

import os
import json
import ast
import numpy as np
import pandas as pd

class Stage03Adapter:
    def __init__(self, config_path="STAGE_04_INTEGRATION/config/paths_config.json"):
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "paths_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.paths = json.load(f)["stage03"]
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

        preds_df = pd.read_csv(self.paths["predictions_path"])
        test_df = pd.read_csv(self.paths["test_split_path"])

        # Merge raw entities from test_df if present
        entities_dict = {}
        if "raw_entities_json" in test_df.columns:
            for _, row in test_df.iterrows():
                nid = row["note_id"]
                try:
                    ents = json.loads(row["raw_entities_json"])
                except Exception:
                    try:
                        ents = ast.literal_eval(row["raw_entities_json"])
                    except Exception:
                        ents = []
                entities_dict[nid] = ents

        # Patient-level aggregation across longitudinal notes
        p_records = []
        for pid, group in preds_df.groupby("patient_id"):
            # Highest risk encounter logic: HIGH > MODERATE > LOW
            has_high = (group["predicted_label"] == "HIGH").any()
            has_mod = (group["predicted_label"] == "MODERATE").any()
            if has_high:
                top_class = "HIGH"
            elif has_mod:
                top_class = "MODERATE"
            else:
                top_class = "LOW"

            mean_low = float(group["prob_LOW"].mean())
            mean_mod = float(group["prob_MODERATE"].mean())
            mean_high = float(group["prob_HIGH"].mean())
            tot = mean_low + mean_mod + mean_high
            p_low = round(mean_low / tot, 4)
            p_mod = round(mean_mod / tot, 4)
            p_high = round(mean_high / tot, 4)

            # Max confidence
            max_conf = float(group["confidence"].max())

            # Collect unique entities across all notes for this patient
            mutations = set()
            drugs = set()
            dosages = set()
            adverse_events = set()
            total_ents = 0

            for nid in group["note_id"]:
                ents = entities_dict.get(nid, [])
                for e in ents:
                    total_ents += 1
                    etype = e.get("label", "")
                    etext = e.get("text", "")
                    if etype == "GENE_MUTATION":
                        mutations.add(etext)
                    elif etype == "DRUG":
                        drugs.add(etext)
                    elif etype == "DOSAGE":
                        dosages.add(etext)
                    elif etype == "ADVERSE_EVENT":
                        adverse_events.add(etext)

            p_records.append({
                "patient_id": pid,
                "nlp_urgency_class": top_class,
                "nlp_prediction": top_class,
                "nlp_low_probability": p_low,
                "nlp_moderate_probability": p_mod,
                "nlp_high_probability": p_high,
                "nlp_risk_probability": p_high,  # P(HIGH)
                "nlp_confidence": round(max_conf, 4),
                "gene_mutations": sorted(list(mutations)),
                "drugs": sorted(list(drugs)),
                "dosages": sorted(list(dosages)),
                "adverse_events": sorted(list(adverse_events)),
                "entity_count": total_ents,
                "num_notes_evaluated": len(group)
            })

        self._predictions_cache = pd.DataFrame(p_records)
        return self._predictions_cache

    def get_patient_prediction(self, patient_id: str) -> dict:
        df = self.load_predictions()
        match = df[df["patient_id"] == patient_id]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

"""Model Agreement Analysis Engine.
STAGE 04 INTEGRATION SUBSYSTEM.
Evaluates cross-modal prediction concordance and discordant risk transitions.
Strictly Read-Only on Upstream Stages.
"""

import pandas as pd
import numpy as np

class ModelAgreementEngine:
    @staticmethod
    def evaluate_agreement(ml_pred: str, dl_pred: str, nlp_pred: str) -> dict:
        """Evaluates pairwise and overall agreement across available predictions."""
        preds = {}
        if ml_pred:
            preds["ML"] = ml_pred
        if dl_pred:
            preds["DL"] = dl_pred
        if nlp_pred:
            preds["NLP"] = nlp_pred

        n_preds = len(preds)
        if n_preds <= 1:
            return {
                "agreement_category": "SINGLE_MODALITY" if n_preds == 1 else "INSUFFICIENT_DATA",
                "ml_vs_dl": "N/A",
                "ml_vs_nlp": "N/A",
                "dl_vs_nlp": "N/A",
                "severe_discordance": False
            }

        unique_preds = set(preds.values())

        # Pairwise
        ml_dl = (ml_pred == dl_pred) if (ml_pred and dl_pred) else "N/A"
        ml_nlp = (ml_pred == nlp_pred) if (ml_pred and nlp_pred) else "N/A"
        dl_nlp = (dl_pred == nlp_pred) if (dl_pred and nlp_pred) else "N/A"

        # Check for severe discordance (LOW vs HIGH)
        severe = False
        if "LOW" in unique_preds and "HIGH" in unique_preds:
            severe = True

        if len(unique_preds) == 1:
            cat = "FULL_AGREEMENT"
        elif severe:
            cat = "DISAGREEMENT"
        else:
            cat = "PARTIAL_AGREEMENT"

        return {
            "agreement_category": cat,
            "ml_vs_dl": "AGREE" if ml_dl is True else ("DISAGREE" if ml_dl is False else "N/A"),
            "ml_vs_nlp": "AGREE" if ml_nlp is True else ("DISAGREE" if ml_nlp is False else "N/A"),
            "dl_vs_nlp": "AGREE" if dl_nlp is True else ("DISAGREE" if dl_nlp is False else "N/A"),
            "severe_discordance": severe
        }

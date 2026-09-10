"""Confidence Normalization Engine.
STAGE 04 INTEGRATION SUBSYSTEM.
Validates probability bounds, verifies probability simplex constraints,
and standardizes confidence metrics across disparate upstream model outputs.
Strictly Read-Only on Upstream Stages.
"""

import numpy as np

class ConfidenceNormalizationEngine:
    @staticmethod
    def validate_probabilities(probs: list) -> bool:
        """Validates that all probabilities are within [0.0, 1.0]."""
        if not probs:
            return False
        for p in probs:
            if p is None or np.isnan(p) or np.isinf(p):
                return False
            if p < -1e-4 or p > 1.0 + 1e-4:
                return False
        return True

    @staticmethod
    def validate_probability_distribution(probs: list, atol: float = 0.02) -> bool:
        """Verifies that a multiclass probability distribution sums approximately to 1.0."""
        if not ConfidenceNormalizationEngine.validate_probabilities(probs):
            return False
        total = float(sum(probs))
        return abs(total - 1.0) <= atol

    @staticmethod
    def normalize_distribution(probs: list) -> list:
        """Projects probabilities onto the standard simplex."""
        clean_probs = [max(0.0, min(1.0, float(p))) for p in probs]
        tot = sum(clean_probs)
        if tot <= 0.0:
            return [1.0 / len(clean_probs)] * len(clean_probs)
        return [round(p / tot, 4) for p in clean_probs]

    @staticmethod
    def compute_entropy_confidence(probs: list) -> float:
        """Computes normalized confidence score (1 - normalized entropy)."""
        dist = np.array(ConfidenceNormalizationEngine.normalize_distribution(probs))
        dist = np.clip(dist, 1e-9, 1.0)
        n_classes = len(dist)
        if n_classes <= 1:
            return 1.0
        entropy = -np.sum(dist * np.log(dist))
        max_entropy = np.log(n_classes)
        conf = 1.0 - (entropy / max_entropy)
        return round(float(np.clip(conf, 0.0, 1.0)), 4)

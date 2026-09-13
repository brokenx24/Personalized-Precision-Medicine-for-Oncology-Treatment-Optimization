"""
Temporal Consistency Validator.
Enforces strict chronological sequence invariants across T0 to T4 longitudinal timepoints.
"""
from typing import Dict, Any, Tuple, List

class TemporalConsistencyValidator:
    REQUIRED_TIMEPOINTS = ["T0", "T1", "T2", "T3", "T4"]

    @classmethod
    def validate(cls, scenario: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        errors = []
        traj = scenario.get("trajectory", {})
        timepoints = traj.get("timepoints", [])

        # Check count
        if len(timepoints) != 5:
            errors.append(f"Trajectory contains {len(timepoints)} timepoints; exactly 5 required (T0-T4).")
            return False, 0.0, errors

        prev_day = -1
        for i, tp in enumerate(timepoints):
            expected_t = cls.REQUIRED_TIMEPOINTS[i]
            actual_t = tp.get("timepoint_id")
            if actual_t != expected_t:
                errors.append(f"Timepoint index {i} mismatch: expected '{expected_t}', found '{actual_t}'.")

            day = tp.get("day_offset", -1)
            # Rule: T0 < T1 < T2 < T3 < T4 with strictly increasing day offset
            if day <= prev_day:
                errors.append(f"Temporal violation at {actual_t}: day {day} is not strictly greater than previous day {prev_day}.")
            prev_day = day

            # Tumor burden must be non-negative
            tb = tp.get("tumor_burden_mm", 0.0)
            if tb < 0.0:
                errors.append(f"Negative tumor burden at {actual_t}: {tb} mm.")

            # ctDNA MAF must be non-negative
            maf = tp.get("ctdna_maf_pct", 0.0)
            if maf < 0.0 or maf > 100.0:
                errors.append(f"ctDNA MAF {maf}% at {actual_t} outside valid 0-100% range.")

        score = max(0.0, 1.0 - (len(errors) * 0.25))
        return len(errors) == 0, round(score, 2), errors

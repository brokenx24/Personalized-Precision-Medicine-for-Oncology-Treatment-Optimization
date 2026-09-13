"""
Clinical Consistency Validator.
Verifies biological and medical invariants between patient demographics, labs, and progress notes.
"""
from typing import Dict, Any, Tuple, List

class ClinicalConsistencyValidator:
    @classmethod
    def validate(cls, scenario: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        errors = []
        patient = scenario.get("patient_profile", {})
        notes = scenario.get("clinical_notes", [])
        traj = scenario.get("trajectory", {})

        # Rule 1: Immutable biological sex
        sex = patient.get("sex")
        if sex not in ["Male", "Female"]:
            errors.append(f"Invalid biological sex: '{sex}'. Must be Male or Female.")

        # Rule 2: Invariant age bounds
        age = patient.get("age", 0)
        if age < 18 or age > 105:
            errors.append(f"Age {age} violates clinical eligibility criteria (18-105).")

        # Rule 3: Clinical note must match structured diagnosis
        c_type = patient.get("cancer_type", "")
        for note in notes:
            text = note.get("clinical_text", "")
            # Check note mentions cancer
            if c_type.split()[0].lower() not in text.lower():
                errors.append(f"Clinical note {note.get('note_id')} fails to reference primary cancer diagnosis '{c_type}'.")

        # Rule 4: Plausible baseline labs
        labs = patient.get("baseline_labs", {})
        creat = labs.get("creatinine_mg_dl", 1.0)
        if creat < 0.2 or creat > 15.0:
            errors.append(f"Creatinine {creat} mg/dL is outside biological plausibility.")

        score = max(0.0, 1.0 - (len(errors) * 0.25))
        return len(errors) == 0, round(score, 2), errors

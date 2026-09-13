"""
Clinical Progress Note Generator.
Produces realistic oncology progress notes strictly grounded in structured trajectory and genomic facts.
"""
from typing import Dict, Any, List

class ClinicalNoteGenerator:
    @classmethod
    def generate_note(
        cls,
        patient_profile: Dict[str, Any],
        trajectory_timepoint: Dict[str, Any],
        mutations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        patient_id = patient_profile["patient_id"]
        t_id = trajectory_timepoint["timepoint_id"]
        days = trajectory_timepoint["day_offset"]
        cancer_type = patient_profile["cancer_type"]
        recist = trajectory_timepoint["recist_status"]
        tumor_mm = trajectory_timepoint["tumor_burden_mm"]
        ctdna = trajectory_timepoint["ctdna_maf_pct"]
        therapy = trajectory_timepoint["treatment_administered"]
        aes = trajectory_timepoint.get("adverse_events", [])
        ae_text = ", ".join(aes) if aes else "None reported"
        primary_mut = mutations[0]["gene"] + " " + mutations[0]["variant"] if mutations else "Genomic profiling pending"

        # Construct strictly grounded clinical progress narrative
        text = (
            f"CLINICAL ONCOLOGY PROGRESS NOTE\n"
            f"Encounter: Day {days} ({t_id}) | Patient ID: {patient_id}\n"
            f"Diagnosis: {cancer_type}, {patient_profile['cancer_stage']}, ECOG Performance Status: {patient_profile['ecog_performance_status']}.\n"
            f"Molecular Subtype: {primary_mut}.\n"
            f"Current Regimen: {therapy}.\n\n"
            f"Assessment & Objective Findings:\n"
            f"Restaging imaging review confirms {recist} with measurable target lesion sum of {tumor_mm} mm. "
            f"Serial circulating tumor DNA (ctDNA) molecular analysis shows allele fraction at {ctdna}% MAF. "
            f"Serum biomarker value is {trajectory_timepoint['primary_biomarker_value']}. "
            f"Toxicities & Adverse Events: Patient reports {ae_text}. "
            f"Plan: Continue simulated clinical decision protocol. Case reviewed for stress-testing precision oncology decision boundaries."
        )

        grounded_entities = [
            {"entity_text": mutations[0]["gene"], "category": "GENE_MUTATION", "verified_in_structured_data": True} if mutations else None,
            {"entity_text": therapy, "category": "DRUG", "verified_in_structured_data": True},
            {"entity_text": f"{tumor_mm} mm", "category": "BIOMARKER", "verified_in_structured_data": True},
            {"entity_text": f"{ctdna}%", "category": "BIOMARKER", "verified_in_structured_data": True}
        ]
        grounded_entities = [e for e in grounded_entities if e is not None]

        return {
            "note_id": f"NOTE-{patient_id}-{t_id}",
            "patient_id": patient_id,
            "timepoint_id": t_id,
            "note_type": "Oncology Consultation / Treatment Follow-up",
            "clinical_text": text,
            "grounded_entities": grounded_entities,
            "disclaimer": "SYNTHETIC CLINICAL PROGRESS NOTE. Research simulation only. Does not contain authentic patient data."
        }

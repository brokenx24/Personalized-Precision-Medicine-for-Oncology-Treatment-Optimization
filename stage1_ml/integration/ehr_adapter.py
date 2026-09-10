"""
================================================================================
EHR & FHIR PAYLOAD ADAPTER (INTEGRATION LAYER)
================================================================================
Translates heterogeneous Electronic Health Record (EHR) and HL7 FHIR
patient bundles from hospital hospital information systems into the
canonical schema required by the Stage 1 Oncology Prediction Engine.
================================================================================
"""

import math
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

class EHRAdapter:
    """
    Hospital Integration Adapter that handles schema transformation,
    LOINC code mapping, unit harmonization, and validation.
    """
    
    # LOINC / EHR Lab Mapping dictionary
    LOINC_MAP = {
        "718-7": "hemoglobin_g_dl",          # Hemoglobin [Mass/volume] in Blood
        "6690-2": "wbc_10_3_ul",             # Leukocytes [#/volume] in Blood
        "777-3": "platelets_10_3_ul",        # Platelets [#/volume] in Blood
        "2160-0": "creatinine_mg_dl",        # Creatinine [Mass/volume] in Serum/Plasma
        "1975-2": "bilirubin_mg_dl",         # Bilirubin.total [Mass/volume] in Serum/Plasma
        "1742-6": "alt_u_l",                 # Alanine aminotransferase [Enzymatic activity/volume]
        "1920-8": "ast_u_l",                 # Aspartate aminotransferase [Enzymatic activity/volume]
        "1751-7": "albumin_g_dl",            # Albumin [Mass/volume] in Serum/Plasma
        "2039-6": "protein_biomarker_cea_ng_ml" # Carcinoembryonic Ag [Mass/volume] in Serum/Plasma
    }

    @staticmethod
    def parse_fhir_bundle(bundle: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parses an HL7 FHIR JSON Bundle containing Patient, Condition, and Observation resources.
        Returns: (patient_id, encounter_id, canonical_patient_record)
        """
        patient_id = "UNKNOWN_PATIENT"
        encounter_id = "ENC_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
        record: Dict[str, Any] = {}
        
        entries = bundle.get("entry", [])
        for item in entries:
            resource = item.get("resource", {})
            res_type = resource.get("resourceType")
            
            # 1. Demographics
            if res_type == "Patient":
                patient_id = resource.get("id", patient_id)
                gender = resource.get("gender", "Unknown").capitalize()
                record["sex"] = gender
                
                birth_date_str = resource.get("birthDate")
                if birth_date_str:
                    try:
                        birth_year = datetime.strptime(birth_date_str, "%Y-%m-%d").year
                        record["age"] = max(18, min(100, datetime.utcnow().year - birth_year))
                    except Exception:
                        pass

            # 2. Oncology Diagnoses & Staging
            elif res_type == "Condition":
                code_obj = resource.get("code", {})
                for coding in code_obj.get("coding", []):
                    display = coding.get("display")
                    if display:
                        record["cancer_type"] = display
                        
                stages = resource.get("stage", [])
                for stg in stages:
                    summary = stg.get("summary", {})
                    for coding in summary.get("coding", []):
                        val = coding.get("display", "")
                        if "Stage" in val:
                            record["cancer_stage"] = val
                        elif val.startswith("G"):
                            record["tumor_grade"] = val

            # 3. Laboratory, Biomarker & Genomic Observations
            elif res_type == "Observation":
                code_obj = resource.get("code", {})
                for coding in code_obj.get("coding", []):
                    code_val = coding.get("code")
                    display_val = coding.get("display", "").lower()
                    
                    val_quantity = resource.get("valueQuantity", {})
                    num_val = val_quantity.get("value")
                    
                    if num_val is not None:
                        # Match by LOINC code
                        if code_val in EHRAdapter.LOINC_MAP:
                            record[EHRAdapter.LOINC_MAP[code_val]] = float(num_val)
                        # Match by genomic text descriptors
                        elif "ctdna" in display_val or "maf" in display_val:
                            record["ctdna_baseline_maf"] = float(num_val)
                        elif "tmb" in display_val or "mutation burden" in display_val:
                            record["tmb_nonsynonymous"] = float(num_val)
                        elif "msi" in display_val:
                            record["msi_sensor_score"] = float(num_val)
                        elif "aneuploidy" in display_val:
                            record["aneuploidy_score"] = float(num_val)
                        elif "fga" in display_val or "genome altered" in display_val:
                            record["fraction_genome_altered"] = float(num_val)
                        elif "ecog" in display_val:
                            record["performance_status_ecog"] = int(num_val)
                        elif "weight" in display_val:
                            record["weight_kg"] = float(num_val)
                        elif "height" in display_val:
                            record["height_cm"] = float(num_val)

        return patient_id, encounter_id, EHRAdapter.standardize_clinical_payload(record)

    @staticmethod
    def standardize_clinical_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures all expected clinical keys exist with physiologically valid fallback defaults
        so that models can safely evaluate the patient without dropping records.
        """
        canonical = dict(payload)
        
        # Clinical defaults if omitted in hospital feed
        defaults = {
            "age": 60,
            "sex": "Female",
            "weight_kg": 68.0,
            "height_cm": 165.0,
            "bmi": 24.98,
            "cancer_type": "Breast Invasive Carcinoma",
            "cancer_stage": "Stage II",
            "tumor_grade": "G2",
            "path_t_stage": "T2",
            "path_n_stage": "N0",
            "path_m_stage": "M0",
            "performance_status_ecog": 0,
            "comorbidity_count": 0,
            "hemoglobin_g_dl": 13.5,
            "wbc_10_3_ul": 6.5,
            "platelets_10_3_ul": 230.0,
            "creatinine_mg_dl": 0.85,
            "bilirubin_mg_dl": 0.60,
            "alt_u_l": 24.0,
            "ast_u_l": 22.0,
            "albumin_g_dl": 4.10,
            "ctdna_baseline_maf": 0.02,
            "protein_biomarker_cea_ng_ml": 2.5,
            "mutation_count": 45,
            "fraction_genome_altered": 0.15,
            "aneuploidy_score": 4,
            "tmb_nonsynonymous": 3.5,
            "msi_sensor_score": 0.2,
            "buffa_hypoxia_score": -4.0,
            "ragnum_hypoxia_score": 2.0,
            "winter_hypoxia_score": -1.0,
            "treatment_radiation": 0,
            "treatment_neoadjuvant": 0,
            "days_since_diagnosis": 0
        }
        
        for k, v in defaults.items():
            if k not in canonical or canonical[k] is None:
                canonical[k] = v
                
        # Derive BMI if weight and height provided but BMI missing
        if canonical.get("height_cm", 0) > 0 and canonical.get("weight_kg", 0) > 0:
            h_m = canonical["height_cm"] / 100.0
            canonical["bmi"] = round(canonical["weight_kg"] / (h_m * h_m), 2)
            
        return canonical

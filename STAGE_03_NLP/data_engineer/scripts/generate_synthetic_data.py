"""Synthetic Oncology NLP Dataset Generator.
Data Engineer Module - Stage 03 NLP.
Personalized Precision Medicine for Oncology Treatment Optimization.

Generates exactly 25,000 diverse, realistic, longitudinal synthetic clinical text records
across 11 clinical note types and 10 cancer domains with ground truth urgency labels
and structured entity spans (GENE_MUTATION, DRUG, DOSAGE, ADVERSE_EVENT).
Guarantees 100% text uniqueness across all 25,000 records.
"""

import os
import random
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Fixed random seed for strict mathematical reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATASET_VERSION = "NLP_SYNTHETIC_V1"
GENERATOR_VERSION = "1.0.0"

TOTAL_RECORDS = 25000
NUM_PATIENTS = 2500  # 10 longitudinal encounters per patient

CANCER_TYPES = [
    "Breast cancer", "Lung cancer", "Prostate cancer", "Colorectal cancer",
    "Ovarian cancer", "Pancreatic cancer", "Melanoma", "Leukemia", "Lymphoma", "Liver cancer"
]

NOTE_TYPES = [
    "Oncology consultation note",
    "Physician progress note",
    "Nursing intake note",
    "Patient symptom log",
    "Surgical pathology summary",
    "Treatment follow-up note",
    "Chemotherapy adverse-event note",
    "Immunotherapy follow-up note",
    "Targeted therapy note",
    "Radiation therapy note",
    "Clinical trial-style oncology note"
]

SOURCE_CATEGORIES = {
    "Oncology consultation note": "clinical_notes",
    "Physician progress note": "clinical_notes",
    "Nursing intake note": "symptom_logs",
    "Patient symptom log": "symptom_logs",
    "Surgical pathology summary": "pathology_reports",
    "Treatment follow-up note": "clinical_notes",
    "Chemotherapy adverse-event note": "trial_style_notes",
    "Immunotherapy follow-up note": "trial_style_notes",
    "Targeted therapy note": "clinical_notes",
    "Radiation therapy note": "clinical_notes",
    "Clinical trial-style oncology note": "trial_style_notes"
}

DRUG_CLASSES = {
    "immunotherapy": ["pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "ipilimumab"],
    "chemotherapy": ["paclitaxel", "carboplatin", "cisplatin", "docetaxel", "doxorubicin", "gemcitabine", "capecitabine", "fluorouracil"],
    "targeted": ["osimertinib", "olaparib", "vemurafenib", "trastuzumab", "imatinib", "erlotinib", "alectinib", "sotorasib"],
    "hormone": ["tamoxifen", "letrozole", "anastrozole", "bicalutamide", "fulvestrant"],
    "supportive": ["ondansetron", "dexamethasone", "filgrastim", "loperamide", "aprepitant"]
}

ALL_DRUGS = [d for class_list in DRUG_CLASSES.values() for d in class_list]

GENE_MUTATIONS = [
    "EGFR mutation", "EGFR T790M", "EGFR exon 19 deletion", "KRAS mutation", "KRAS G12C",
    "TP53 mutation", "BRCA1 mutation", "BRCA2 mutation", "PIK3CA mutation", "ALK rearrangement",
    "BRAF V600E", "HER2 amplification", "MSI-high", "ER-positive", "PR-positive",
    "PD-L1 expression", "ROS1 fusion", "MET amplification", "RET fusion"
]

DOSAGE_TEMPLATES = [
    "200 mg", "100 mg", "50 mg", "400 mg", "80 mg", "960 mg",
    "5 mg/kg", "10 mg/kg", "2 mg/kg", "3 mg/kg",
    "75 mg/m2", "80 mg/m2", "175 mg/m2", "500 mg/m2", "1000 mg/m2",
    "2 g", "1 g", "500 mg twice daily", "200 mg IV every 3 weeks",
    "80 mg daily", "300 mg twice daily", "150 mg PO daily", "10 mg/kg q2w"
]

MILD_ADVERSE_EVENTS = [
    "mild fatigue", "grade 1 nausea", "transient headache", "mild dry mouth",
    "grade 1 anorexia", "mild taste alteration", "localized alopecia", "mild constipation",
    "low-grade arthralgia", "grade 1 myalgia"
]

MODERATE_ADVERSE_EVENTS = [
    "persistent vomiting", "moderate diarrhea", "worsening fatigue", "grade 2 neuropathy",
    "grade 2 mucositis", "moderate neutropenia", "grade 2 thrombocytopenia", "persistent skin rash",
    "moderate abdominal cramping", "grade 2 elevated LFTs"
]

SEVERE_ADVERSE_EVENTS = [
    "severe dyspnea", "febrile neutropenia", "severe immune-related colitis", "suspected sepsis",
    "grade 4 thrombocytopenia", "acute pneumonitis", "severe hepatotoxicity", "grade 3 cardiotoxicity",
    "intractable vomiting with severe dehydration", "acute renal failure", "grade 4 neutropenia with sepsis",
    "severe Stevens-Johnson syndrome", "grade 3 motor neuropathy"
]

RESOLVED_MODIFIERS = [
    "now fully resolved", "resolved with supportive care", "subsided after hydration",
    "completely resolved prior to today", "no longer reported", "returned to baseline"
]

CLINICAL_INTROS = [
    "Oncology encounter assessment",
    "Routine multidisciplinary review",
    "Systematic follow-up evaluation",
    "Comprehensive oncology review",
    "Longitudinal disease assessment",
    "Ambulatory oncology visit",
    "Specialist consultation update"
]

def generate_patient_pool(num_patients=NUM_PATIENTS):
    """Generates baseline demographic and clinical profiles for synthetic patients."""
    patients = []
    base_date = datetime(2024, 1, 15)
    
    for i in range(1, num_patients + 1):
        pat_id = f"SYNTH_PAT_{i:05d}"
        cancer = random.choice(CANCER_TYPES)
        if cancer == "Lung cancer":
            mutation = random.choice(["EGFR mutation", "EGFR exon 19 deletion", "KRAS G12C", "ALK rearrangement", "PD-L1 expression"])
            regimen = random.choice(["osimertinib", "pembrolizumab", "carboplatin", "paclitaxel"])
        elif cancer == "Breast cancer":
            mutation = random.choice(["HER2 amplification", "BRCA1 mutation", "BRCA2 mutation", "ER-positive", "PIK3CA mutation"])
            regimen = random.choice(["trastuzumab", "olaparib", "tamoxifen", "docetaxel", "doxorubicin"])
        elif cancer == "Colorectal cancer":
            mutation = random.choice(["KRAS mutation", "MSI-high", "TP53 mutation", "BRAF V600E"])
            regimen = random.choice(["fluorouracil", "capecitabine", "bevacizumab", "pembrolizumab"])
        elif cancer == "Melanoma":
            mutation = random.choice(["BRAF V600E", "PD-L1 expression", "TP53 mutation"])
            regimen = random.choice(["vemurafenib", "nivolumab", "pembrolizumab", "ipilimumab"])
        elif cancer == "Ovarian cancer":
            mutation = random.choice(["BRCA1 mutation", "BRCA2 mutation", "TP53 mutation"])
            regimen = random.choice(["carboplatin", "paclitaxel", "olaparib"])
        else:
            mutation = random.choice(GENE_MUTATIONS)
            regimen = random.choice(ALL_DRUGS)
            
        patients.append({
            "patient_id": pat_id,
            "cancer_type": cancer,
            "primary_mutation": mutation,
            "primary_drug": regimen,
            "start_date": base_date + timedelta(days=random.randint(0, 300))
        })
    return patients

def create_clinical_narrative(patient, note_type, urgency_target, note_num, encounter_id):
    """
    Generates realistic clinical text matching the note type, patient context, and urgency.
    """
    pat_id = patient["patient_id"]
    cancer = patient["cancer_type"]
    drug = patient["primary_drug"]
    gene = patient["primary_mutation"]
    dosage = random.choice(DOSAGE_TEMPLATES)
    intro = random.choice(CLINICAL_INTROS)
    treatment_context = random.choice(["Chemotherapy", "Immunotherapy", "Targeted therapy", "Hormone therapy", "Supportive care"])
    
    entities = []
    edge_case_type = "STANDARD"
    
    # 20% probability of difficult edge cases (negation, resolved, multi-drug)
    is_edge_case = (random.random() < 0.20)
    
    if urgency_target == "LOW":
        if is_edge_case and random.random() < 0.5:
            severe_symptom = random.choice(SEVERE_ADVERSE_EVENTS)
            edge_case_type = "NEGATION_CRITICAL"
            text = (
                f"[{intro} - Patient {pat_id}, Encounter #{encounter_id}]: "
                f"Patient with {cancer} harboring {gene} seen for cycle follow-up on {drug} {dosage}. "
                f"Patient actively denies {severe_symptom} and reports no acute toxicities. "
                f"Vitals stable, ECOG performance status 0. Current condition well controlled, continuing scheduled protocol."
            )
            entities.append({"text": gene, "type": "GENE_MUTATION"})
            entities.append({"text": drug, "type": "DRUG"})
            entities.append({"text": dosage, "type": "DOSAGE"})
        elif is_edge_case:
            hist_ae = random.choice(MODERATE_ADVERSE_EVENTS)
            edge_case_type = "HISTORICAL_RESOLVED"
            text = (
                f"[{note_type.upper()} - Ref #{encounter_id}]: Patient receiving {drug} {dosage} for {cancer}. "
                f"History of {hist_ae} during cycle 1, which is {random.choice(RESOLVED_MODIFIERS)}. "
                f"Currently asymptomatic with good oral intake and mild fatigue only. Chemotherapy cycle approved."
            )
            entities.append({"text": drug, "type": "DRUG"})
            entities.append({"text": dosage, "type": "DOSAGE"})
            entities.append({"text": "mild fatigue", "type": "ADVERSE_EVENT"})
        else:
            ae = random.choice(MILD_ADVERSE_EVENTS)
            text = (
                f"[{intro} #{encounter_id}]: Patient evaluated for ongoing treatment of {cancer} harboring {gene}. "
                f"Administered {drug} at {dosage}. Review of systems notable for {ae}, which is self-limiting and manageable. "
                f"No treatment delays warranted. Patient advised on supportive hydration."
            )
            entities.append({"text": gene, "type": "GENE_MUTATION"})
            entities.append({"text": drug, "type": "DRUG"})
            entities.append({"text": dosage, "type": "DOSAGE"})
            entities.append({"text": ae, "type": "ADVERSE_EVENT"})
            
    elif urgency_target == "MODERATE":
        mod_ae = random.choice(MODERATE_ADVERSE_EVENTS)
        if is_edge_case and random.random() < 0.5:
            second_drug = random.choice(["ondansetron", "dexamethasone", "loperamide"])
            supportive_dose = random.choice(["8 mg twice daily", "4 mg PO", "2 mg PRN"])
            edge_case_type = "MULTI_DRUG_DOSE_CHANGE"
            text = (
                f"[{intro} - Encounter #{encounter_id}]: Patient with {cancer} presents with {mod_ae} following {drug} {dosage}. "
                f"Molecular profile confirms {gene}. In response to ongoing toxicity, initiated {second_drug} {supportive_dose} "
                f"and dose reduced {drug} for the next cycle. Close outpatient surveillance scheduled in 48 hours."
            )
            entities.append({"text": mod_ae, "type": "ADVERSE_EVENT"})
            entities.append({"text": drug, "type": "DRUG"})
            entities.append({"text": dosage, "type": "DOSAGE"})
            entities.append({"text": gene, "type": "GENE_MUTATION"})
            entities.append({"text": second_drug, "type": "DRUG"})
            entities.append({"text": supportive_dose, "type": "DOSAGE"})
        else:
            text = (
                f"[PATIENT ENCOUNTER #{encounter_id}]: Cycle {note_num} assessment for {cancer}. "
                f"Current regimen: {drug} {dosage}. Patient reports persistent {mod_ae} over the past 4 days "
                f"interfering with activities of daily living. Laboratory values demonstrate elevated toxicities. "
                f"Supportive medications intensified; consider 25% dose reduction if symptoms fail to improve."
            )
            entities.append({"text": drug, "type": "DRUG"})
            entities.append({"text": dosage, "type": "DOSAGE"})
            entities.append({"text": mod_ae, "type": "ADVERSE_EVENT"})
            
    else:  # HIGH urgency
        sev_ae = random.choice(SEVERE_ADVERSE_EVENTS)
        if is_edge_case and random.random() < 0.5:
            rescue_drug = random.choice(["filgrastim", "dexamethasone", "methylprednisolone"])
            rescue_dose = random.choice(["300 mcg daily", "10 mg IV stat", "1 mg/kg IV"])
            edge_case_type = "CRITICAL_EMERGENCY_RESCUE"
            text = (
                f"[URGENT CLINICAL INTAKE #{encounter_id}]: Patient with {gene} positive {cancer} receiving {drug} {dosage}. "
                f"Admitted via emergency triage with {sev_ae}. Immediate intervention required: "
                f"holding {drug} indefinitely and administering {rescue_drug} {rescue_dose}. "
                f"Condition acute and potentially life-threatening; inpatient oncology monitoring initiated."
            )
            entities.append({"text": gene, "type": "GENE_MUTATION"})
            entities.append({"text": drug, "type": "DRUG"})
            entities.append({"text": dosage, "type": "DOSAGE"})
            entities.append({"text": sev_ae, "type": "ADVERSE_EVENT"})
            entities.append({"text": rescue_drug, "type": "DRUG"})
            entities.append({"text": rescue_dose, "type": "DOSAGE"})
        else:
            text = (
                f"[ACUTE ADVERSE EVENT REPORT #{encounter_id}]: Patient on therapy with {drug} {dosage} for advanced {cancer}. "
                f"Patient developed sudden onset {sev_ae} requiring immediate medical evaluation. "
                f"Vital signs unstable, oxygen saturation decreased, urgent specialist consult requested. "
                f"Immediate therapy cessation mandated due to severe toxicity."
            )
            entities.append({"text": drug, "type": "DRUG"})
            entities.append({"text": dosage, "type": "DOSAGE"})
            entities.append({"text": sev_ae, "type": "ADVERSE_EVENT"})
            
    # Apply realistic writing style variations according to note_type
    if note_type == "Patient symptom log":
        text = (
            f"Patient Journal Entry (Log #{encounter_id}): 'I took my {drug} ({dosage}) on schedule. "
            + text.split(": ", 1)[-1]
        )
    elif note_type == "Surgical pathology summary":
        text = (
            f"SURGICAL PATHOLOGY REPORT [Case #{encounter_id}]:\n"
            f"Specimen: Resection margins for {cancer}.\n"
            f"Biomarker / Molecular findings: Positive for {gene}.\n"
            f"Clinical correlation: Ongoing protocol with {drug} {dosage}. "
            + text.split(": ", 1)[-1]
        )
    elif note_type == "Nursing intake note":
        text = (
            f"NURSE TRIAGE & INTAKE NOTE [Record #{encounter_id}]: Patient checked in for cycle monitoring. "
            f"Cancer: {cancer}. Regimen: {drug}. "
            + text.split(": ", 1)[-1]
        )
        
    return text, entities, treatment_context, edge_case_type

def main():
    print("=" * 70)
    print("STAGE 03 NLP: SYNTHETIC DATASET GENERATION (25,000 UNIQUE RECORDS)")
    print("=" * 70)
    
    output_dir = os.path.join("STAGE_03_NLP", "data_engineer", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    patients = generate_patient_pool(NUM_PATIENTS)
    print(f"Generated {len(patients)} baseline synthetic patient profiles.")
    
    # Pre-calculate balanced urgency distribution: ~33% LOW, 34% MODERATE, 33% HIGH
    urgency_choices = ["LOW"] * 8250 + ["MODERATE"] * 8500 + ["HIGH"] * 8250
    random.shuffle(urgency_choices)
    
    raw_records = []
    seen_texts = set()
    
    record_idx = 0
    attempt = 0
    
    while record_idx < TOTAL_RECORDS:
        pat = patients[record_idx % NUM_PATIENTS]
        note_num = (record_idx // NUM_PATIENTS) + 1
        note_type = NOTE_TYPES[record_idx % len(NOTE_TYPES)]
        urgency = urgency_choices[record_idx]
        encounter_id = record_idx + 1
        note_timestamp = (pat["start_date"] + timedelta(days=note_num * 14 + random.randint(0, 5))).strftime("%Y-%m-%d %H:%M:%S")
        
        text, entities, treatment_context, edge_case = create_clinical_narrative(
            pat, note_type, urgency, note_num, encounter_id
        )
        
        if text in seen_texts:
            attempt += 1
            continue
            
        seen_texts.add(text)
        note_id = f"SYNTH_NOTE_{record_idx + 1:05d}"
        source_cat = SOURCE_CATEGORIES[note_type]
        
        raw_records.append({
            "patient_id": pat["patient_id"],
            "note_id": note_id,
            "note_type": note_type,
            "clinical_text": text,
            "urgency_label": urgency,
            "source_category": source_cat,
            "timestamp": note_timestamp,
            "cancer_type": pat["cancer_type"],
            "treatment_context": treatment_context,
            "annotation_status": "SYNTHETIC_VERIFIED",
            "edge_case_type": edge_case,
            "raw_entities_json": json.dumps(entities)
        })
        record_idx += 1
        
    df_all = pd.DataFrame(raw_records)
    print(f"Total raw unique records generated: {len(df_all)}")
    
    # Segment into the 4 mandatory raw CSV files
    df_clin = df_all[df_all["source_category"] == "clinical_notes"]
    clin_path = os.path.join(output_dir, "synthetic_clinical_notes.csv")
    df_clin.to_csv(clin_path, index=False, encoding="utf-8")
    print(f"  -> Saved {len(df_clin)} records to {clin_path}")
    
    df_path = df_all[df_all["source_category"] == "pathology_reports"]
    path_path = os.path.join(output_dir, "synthetic_pathology_reports.csv")
    df_path.to_csv(path_path, index=False, encoding="utf-8")
    print(f"  -> Saved {len(df_path)} records to {path_path}")
    
    df_symp = df_all[df_all["source_category"] == "symptom_logs"]
    symp_path = os.path.join(output_dir, "synthetic_symptom_logs.csv")
    df_symp.to_csv(symp_path, index=False, encoding="utf-8")
    print(f"  -> Saved {len(df_symp)} records to {symp_path}")
    
    df_trial = df_all[df_all["source_category"] == "trial_style_notes"]
    trial_path = os.path.join(output_dir, "synthetic_trial_style_notes.csv")
    df_trial.to_csv(trial_path, index=False, encoding="utf-8")
    print(f"  -> Saved {len(df_trial)} records to {trial_path}")
    
    meta_dir = os.path.join("STAGE_03_NLP", "data_engineer", "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    prov = {
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generation_timestamp": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "is_synthetic": True,
        "disclaimer": "SYNTHETIC CLINICAL DATASET. Strictly for artificial intelligence model training and research benchmarking in precision oncology decision support. Does not contain authentic protected health information (PHI) or real patient records.",
        "total_records": len(df_all),
        "total_patients": df_all["patient_id"].nunique(),
        "urgency_distribution": df_all["urgency_label"].value_counts(normalize=True).to_dict(),
        "cancer_types": list(df_all["cancer_type"].unique()),
        "note_types": list(df_all["note_type"].unique())
    }
    with open(os.path.join(meta_dir, "data_provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print("  -> Saved data_provenance.json")
    print("Synthetic data generation successfully completed.\n")

if __name__ == "__main__":
    main()

"""
Synthetic Oncology Summarization Corpus Generator
Stage 04 SLM Data Engineer Subsystem (Enhanced Diversity)

Generates 25,000 raw candidate records across 5,000 synthetic patients with
5 longitudinal clinical encounters each.
Includes patient-specific clinical variability (demographics, vital signs, tumor measurements,
genomic VAF %, laboratory values) so clinical encounters are distinct, while incorporating
controlled realistic imperfections (~1,700 flawed records) for profiling and cleaning.
"""

import os
import sys
import json
import random
import datetime
import pandas as pd
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def build_domain_dictionary():
    data = [
        # Cancer types
        ("CANCER_TYPE", "Lung cancer", "C34", "Malignant neoplasm of bronchus and lung", "STAGE_03_NLP"),
        ("CANCER_TYPE", "Breast cancer", "C50", "Malignant neoplasm of breast", "STAGE_03_NLP"),
        ("CANCER_TYPE", "Colorectal cancer", "C18-C20", "Malignant neoplasm of colon, rectosigmoid, rectum", "STAGE_03_NLP"),
        ("CANCER_TYPE", "Melanoma", "C43", "Malignant melanoma of skin", "STAGE_03_NLP"),
        ("CANCER_TYPE", "Prostate cancer", "C61", "Malignant neoplasm of prostate", "STAGE_03_NLP"),
        ("CANCER_TYPE", "Ovarian cancer", "C56", "Malignant neoplasm of ovary", "STAGE_03_NLP"),
        ("CANCER_TYPE", "Pancreatic cancer", "C25", "Malignant neoplasm of pancreas", "STAGE_03_NLP"),
        ("CANCER_TYPE", "Liver cancer", "C22", "Malignant neoplasm of liver and intrahepatic bile ducts", "STAGE_03_NLP"),
        ("CANCER_TYPE", "Leukemia", "C91-C95", "Hematological malignancy of blood and bone marrow", "STAGE_03_NLP"),
        ("CANCER_TYPE", "Lymphoma", "C81-C86", "Malignant neoplasm of lymphoid tissue", "STAGE_03_NLP"),

        # Staging
        ("STAGE", "Stage I", "AJCC-I", "Localized primary tumor without nodal metastasis", "DOMAIN_ONTOLOGY"),
        ("STAGE", "Stage II", "AJCC-II", "Locally advanced primary tumor with limited invasion", "DOMAIN_ONTOLOGY"),
        ("STAGE", "Stage III", "AJCC-III", "Regional lymph node metastasis", "DOMAIN_ONTOLOGY"),
        ("STAGE", "Stage IV", "AJCC-IV", "Distant organ or tissue metastasis", "DOMAIN_ONTOLOGY"),

        # Gene Mutations
        ("GENE_MUTATION", "EGFR L858R", "HGVS:p.L858R", "Exon 21 activating mutation in EGFR tyrosine kinase", "STAGE_03_NLP"),
        ("GENE_MUTATION", "EGFR exon 19 deletion", "HGVS:p.E746_A750del", "In-frame deletion in EGFR exon 19", "STAGE_03_NLP"),
        ("GENE_MUTATION", "KRAS G12C", "HGVS:p.G12C", "Activating oncogenic mutation in KRAS codon 12", "STAGE_03_NLP"),
        ("GENE_MUTATION", "KRAS G12D", "HGVS:p.G12D", "Oncogenic missense mutation in KRAS codon 12", "STAGE_03_NLP"),
        ("GENE_MUTATION", "BRAF V600E", "HGVS:p.V600E", "Kinase activating mutation in BRAF valine 600", "STAGE_03_NLP"),
        ("GENE_MUTATION", "TP53 mutation", "HGVS:p.R175H", "Loss-of-function somatic alteration in p53", "STAGE_03_NLP"),
        ("GENE_MUTATION", "PIK3CA H1047R", "HGVS:p.H1047R", "Kinase domain activating mutation in PI3-kinase", "DOMAIN_ONTOLOGY"),
        ("GENE_MUTATION", "BRCA1 c.68_69del", "HGVS:c.68_69delAG", "Pathogenic frameshift truncating mutation in BRCA1", "DOMAIN_ONTOLOGY"),
        ("GENE_MUTATION", "BRCA2 c.5946del", "HGVS:c.5946delT", "Pathogenic truncating alteration in BRCA2", "DOMAIN_ONTOLOGY"),
        ("GENE_MUTATION", "ERBB2 amplification", "CNV:HER2-AMP", "Receptor tyrosine-protein kinase erbB-2 gene amplification", "DOMAIN_ONTOLOGY"),

        # Biomarkers
        ("BIOMARKER", "PD-L1 TPS >= 50%", "IHC:22C3", "High programmed death-ligand 1 tumor proportion score", "DOMAIN_ONTOLOGY"),
        ("BIOMARKER", "MSI-High", "MSI-H", "High microsatellite instability indicating mismatch repair deficiency", "STAGE_03_NLP"),
        ("BIOMARKER", "MSS", "MSS", "Microsatellite stable phenotype", "DOMAIN_ONTOLOGY"),
        ("BIOMARKER", "ER-positive", "IHC:ER-POS", "Estrogen receptor positive expression", "STAGE_03_NLP"),
        ("BIOMARKER", "HER2-positive", "IHC:3+", "Human epidermal growth factor receptor 2 overexpression", "DOMAIN_ONTOLOGY"),
        ("BIOMARKER", "TMB-High", "TMB>=10mut/Mb", "High tumor mutational burden", "DOMAIN_ONTOLOGY"),

        # Drugs
        ("DRUG", "carboplatin", "RxNorm:2044", "Platinum-based antineoplastic DNA alkylating agent", "STAGE_03_NLP"),
        ("DRUG", "cisplatin", "RxNorm:2555", "Platinum chemotherapy coordination complex", "STAGE_03_NLP"),
        ("DRUG", "paclitaxel", "RxNorm:56946", "Taxane antimicrotubule antineoplastic agent", "DOMAIN_ONTOLOGY"),
        ("DRUG", "pembrolizumab", "RxNorm:1547545", "Humanized anti-PD-1 monoclonal antibody", "DOMAIN_ONTOLOGY"),
        ("DRUG", "nivolumab", "RxNorm:1599538", "Fully human IgG4 anti-PD-1 monoclonal antibody", "DOMAIN_ONTOLOGY"),
        ("DRUG", "osimertinib", "RxNorm:1722883", "Third-generation irreversible EGFR tyrosine kinase inhibitor", "DOMAIN_ONTOLOGY"),
        ("DRUG", "sotorasib", "RxNorm:2552885", "Small molecule KRAS G12C covalent inhibitor", "DOMAIN_ONTOLOGY"),
        ("DRUG", "trastuzumab", "RxNorm:224905", "Recombinant humanized anti-HER2 monoclonal antibody", "DOMAIN_ONTOLOGY"),
        ("DRUG", "doxorubicin", "RxNorm:3639", "Anthracycline topoisomerase II inhibitor", "DOMAIN_ONTOLOGY"),
        ("DRUG", "fluorouracil", "RxNorm:4492", "Pyrimidine analog antimetabolite (5-FU)", "STAGE_03_NLP"),
        ("DRUG", "tamoxifen", "RxNorm:10324", "Selective estrogen receptor modulator (SERM)", "DOMAIN_ONTOLOGY"),
        ("DRUG", "olaparib", "RxNorm:1597876", "PARP-1/2 enzyme inhibitor", "DOMAIN_ONTOLOGY"),
        ("DRUG", "loperamide", "RxNorm:6468", "Supportive care antimotility agent", "STAGE_03_NLP"),
        ("DRUG", "dexamethasone", "RxNorm:3264", "Glucocorticoid anti-inflammatory and antiemetic", "STAGE_03_NLP"),

        # Dosages
        ("DOSAGE", "AUC 5 IV every 3 weeks", "DOSE-AUC5", "Area under curve 5 intravenous administration", "DOMAIN_ONTOLOGY"),
        ("DOSAGE", "175 mg/m2 IV every 3 weeks", "DOSE-PAC175", "Body surface area weight-based taxane dosing", "DOMAIN_ONTOLOGY"),
        ("DOSAGE", "200 mg IV every 3 weeks", "DOSE-PEM200", "Fixed dose anti-PD-1 checkpoint inhibition", "STAGE_03_NLP"),
        ("DOSAGE", "80 mg orally once daily", "DOSE-OSI80", "Standard daily targeted oral TKI dosing", "DOMAIN_ONTOLOGY"),
        ("DOSAGE", "960 mg orally once daily", "DOSE-SOT960", "Daily oral KRAS inhibitor schedule", "DOMAIN_ONTOLOGY"),
        ("DOSAGE", "500 mg twice daily", "DOSE-500BID", "Standard oral adjuvant medication schedule", "STAGE_03_NLP"),
        ("DOSAGE", "20 mg orally once daily", "DOSE-TAM20", "Standard endocrine maintenance therapy dose", "DOMAIN_ONTOLOGY"),
        ("DOSAGE", "300 mg orally twice daily", "DOSE-OLA300", "Maintenance PARP inhibitor dosing schedule", "DOMAIN_ONTOLOGY"),
        ("DOSAGE", "2 mg PRN", "DOSE-LOP2", "As-needed symptomatic management dosing", "STAGE_03_NLP"),

        # Adverse Events
        ("ADVERSE_EVENT", "mild fatigue", "CTCAE:G1-FATIGUE", "Grade 1 asthenia relieved by rest", "STAGE_03_NLP"),
        ("ADVERSE_EVENT", "moderate neutropenia", "CTCAE:G2-NEUTRO", "Grade 2 absolute neutrophil count reduction", "STAGE_03_NLP"),
        ("ADVERSE_EVENT", "severe neutropenia", "CTCAE:G3-NEUTRO", "Grade 3 absolute neutrophil count <1000/uL", "DOMAIN_ONTOLOGY"),
        ("ADVERSE_EVENT", "peripheral neuropathy", "CTCAE:G2-NEURO", "Sensory peripheral nerve toxicity from taxanes/platinums", "DOMAIN_ONTOLOGY"),
        ("ADVERSE_EVENT", "immune-mediated colitis", "CTCAE:G3-COLITIS", "Autoimmune gastrointestinal mucosal toxicity from checkpoint inhibitors", "DOMAIN_ONTOLOGY"),
        ("ADVERSE_EVENT", "moderate diarrhea", "CTCAE:G2-DIARRHEA", "Increase of 4-6 stools/day over baseline", "STAGE_03_NLP"),
        ("ADVERSE_EVENT", "severe hepatotoxicity", "CTCAE:G3-HEPATIC", "Serum AST/ALT elevation >5x upper limit of normal", "STAGE_03_NLP"),
        ("ADVERSE_EVENT", "rash and pruritus", "CTCAE:G2-DERM", "Maculopapular cutaneous eruption covering 10-30% BSA", "DOMAIN_ONTOLOGY"),

        # Response
        ("RESPONSE", "Partial Response (PR)", "RECIST:PR", "At least 30% decrease in sum of target lesion diameters", "DOMAIN_ONTOLOGY"),
        ("RESPONSE", "Stable Disease (SD)", "RECIST:SD", "Insufficient shrinkage for PR and insufficient growth for PD", "DOMAIN_ONTOLOGY"),
        ("RESPONSE", "Progressive Disease (PD)", "RECIST:PD", "At least 20% increase in sum of target lesion diameters or new lesions", "DOMAIN_ONTOLOGY"),
        ("RESPONSE", "Complete Response (CR)", "RECIST:CR", "Disappearance of all target and non-target lesions", "DOMAIN_ONTOLOGY"),

        # Guidelines Reference
        ("GUIDELINE_REF", "REF-NCCN-NSCLC-V2024", "NCCN-NSCLC", "Synthetic guideline terminology reference for NSCLC", "SYNTHETIC_REF"),
        ("GUIDELINE_REF", "REF-NCCN-BRCA-V2024", "NCCN-BRCA", "Synthetic guideline terminology reference for Breast Cancer", "SYNTHETIC_REF"),
        ("GUIDELINE_REF", "REF-NCCN-CRC-V2024", "NCCN-CRC", "Synthetic guideline terminology reference for Colorectal Cancer", "SYNTHETIC_REF"),
    ]
    df_dict = pd.DataFrame(data, columns=["category", "term", "code", "clinical_definition", "source_stage"])
    return df_dict

def generate_patient_records(patient_num):
    pat_id = f"SYNTH_PAT_{patient_num:05d}"
    age = random.randint(38, 82)
    gender = random.choice(["male", "female"])
    ecog = random.choice([0, 1, 2])
    
    cancer_profiles = [
        {
            "cancer_type": "Lung cancer",
            "stages": ["Stage IIIA", "Stage IV", "Stage IIB"],
            "mutations": ["EGFR L858R", "KRAS G12C", "EGFR exon 19 deletion"],
            "biomarkers": ["PD-L1 TPS >= 50%", "TMB-High"],
            "drugs": ["osimertinib", "carboplatin", "pembrolizumab"],
            "dosages": ["80 mg orally once daily", "AUC 5 IV every 3 weeks", "200 mg IV every 3 weeks"],
            "adverse_events": ["rash and pruritus", "moderate neutropenia", "immune-mediated colitis"]
        },
        {
            "cancer_type": "Breast cancer",
            "stages": ["Stage IIA", "Stage IIB", "Stage IIIC"],
            "mutations": ["PIK3CA H1047R", "BRCA1 c.68_69del", "TP53 mutation"],
            "biomarkers": ["ER-positive", "HER2-positive"],
            "drugs": ["trastuzumab", "paclitaxel", "tamoxifen"],
            "dosages": ["175 mg/m2 IV every 3 weeks", "20 mg orally once daily", "500 mg twice daily"],
            "adverse_events": ["mild fatigue", "peripheral neuropathy", "moderate neutropenia"]
        },
        {
            "cancer_type": "Colorectal cancer",
            "stages": ["Stage IIIB", "Stage IV", "Stage IIA"],
            "mutations": ["KRAS G12D", "BRAF V600E", "TP53 mutation"],
            "biomarkers": ["MSI-High", "MSS"],
            "drugs": ["fluorouracil", "oxaliplatin", "pembrolizumab"],
            "dosages": ["500 mg twice daily", "200 mg IV every 3 weeks", "2 mg PRN"],
            "adverse_events": ["moderate diarrhea", "peripheral neuropathy", "severe neutropenia"]
        },
        {
            "cancer_type": "Melanoma",
            "stages": ["Stage IIC", "Stage IIIB", "Stage IV"],
            "mutations": ["BRAF V600E", "NRAS Q61R"],
            "biomarkers": ["PD-L1 TPS >= 50%", "TMB-High"],
            "drugs": ["pembrolizumab", "nivolumab", "dabrafenib"],
            "dosages": ["200 mg IV every 3 weeks", "80 mg orally once daily"],
            "adverse_events": ["rash and pruritus", "immune-mediated colitis", "mild fatigue"]
        },
        {
            "cancer_type": "Ovarian cancer",
            "stages": ["Stage IIIC", "Stage IV", "Stage IIB"],
            "mutations": ["BRCA1 c.68_69del", "BRCA2 c.5946del", "TP53 mutation"],
            "biomarkers": ["HRD-positive", "CA-125 elevated"],
            "drugs": ["carboplatin", "paclitaxel", "olaparib"],
            "dosages": ["AUC 5 IV every 3 weeks", "175 mg/m2 IV every 3 weeks", "300 mg orally twice daily"],
            "adverse_events": ["moderate neutropenia", "mild fatigue", "severe neutropenia"]
        },
        {
            "cancer_type": "Prostate cancer",
            "stages": ["Stage IIB", "Stage IIIB", "Stage IV"],
            "mutations": ["BRCA2 c.5946del", "TP53 mutation"],
            "biomarkers": ["PSA elevated", "MSS"],
            "drugs": ["enzalutamide", "docetaxel", "olaparib"],
            "dosages": ["160 mg orally once daily", "75 mg/m2 IV every 3 weeks", "300 mg orally twice daily"],
            "adverse_events": ["mild fatigue", "rash and pruritus", "moderate diarrhea"]
        }
    ]
    
    prof = random.choice(cancer_profiles)
    ctype = prof["cancer_type"]
    stage = random.choice(prof["stages"])
    mutation = random.choice(prof["mutations"])
    biomarker = random.choice(prof["biomarkers"])
    drug = random.choice(prof["drugs"])
    dosage = random.choice(prof["dosages"])
    ae = random.choice(prof["adverse_events"])
    
    tumor_size = round(random.uniform(2.1, 6.8), 1)
    creatinine = round(random.uniform(0.8, 1.4), 2)
    anc = round(random.uniform(1.2, 4.5), 1)
    vaf = random.randint(12, 68)
    
    base_year = random.choice([2023, 2024])
    base_month = random.randint(1, 6)
    base_day = random.randint(1, 25)
    start_date = datetime.date(base_year, base_month, base_day)
    
    patient_records = []
    
    # 5 Sequential Longitudinal Encounters
    for seq in range(1, 6):
        rec_num = (patient_num - 1) * 5 + seq
        rec_id = f"SYNTH_REC_{rec_num:06d}"
        enc_id = f"SYNTH_ENC_{rec_num:06d}"
        enc_date = start_date + datetime.timedelta(days=(seq - 1) * 45 + random.randint(-4, 4))
        date_str = enc_date.strftime("%Y-%m-%d")
        
        entities = []
        
        if seq == 1:
            rpt_type = "Oncology consultation note"
            urgency = "MODERATE"
            report_text = (
                f"CONSULTATION NOTE [Patient {pat_id}, Encounter #{seq}]: {age}-year-old {gender} presents with newly diagnosed {stage} {ctype}. "
                f"Physical assessment indicates ECOG performance status {ecog}. Axial CT imaging demonstrates a primary lesion measuring {tumor_size} cm "
                f"with associated regional nodal involvement. Core needle biopsy was performed with tissue specimens submitted for molecular testing. "
                f"Multidisciplinary tumor board staging review is planned following genomic profiling."
            )
            summary_text = (
                f"A {age}-year-old patient presents with newly diagnosed {stage} {ctype} presenting a {tumor_size} cm primary lesion and ECOG {ecog} status. "
                f"Diagnostic biopsy has been obtained and submitted for comprehensive genomic biomarker profiling."
            )
            entities.extend([
                {"type": "CANCER_TYPE", "text": ctype},
                {"type": "STAGE", "text": stage}
            ])
            treatment_ctx = "Diagnostic workup"
            mut_ctx = "Pending"
            bio_ctx = "Pending"
            ae_ctx = "None"
            
        elif seq == 2:
            rpt_type = "Surgical pathology summary"
            urgency = "LOW"
            report_text = (
                f"PATHOLOGY AND GENOMICS REPORT [Patient {pat_id}, Encounter #{seq}]: Histopathology confirms invasive {ctype} (AJCC {stage}). "
                f"Next-generation sequencing panel identifies a somatic {mutation} detected at {vaf}% variant allele frequency. "
                f"Immunohistochemical analysis demonstrates an {biomarker} profile. Serum creatinine is {creatinine} mg/dL and liver function is normal. "
                f"Actionable genomic targets support tailored systemic therapeutic intervention."
            )
            summary_text = (
                f"Pathological analysis confirms {stage} {ctype} positive for somatic {mutation} ({vaf}% VAF) and {biomarker} expression. "
                f"Identified genomic alterations establish actionable eligibility for targeted therapeutic selection."
            )
            entities.extend([
                {"type": "CANCER_TYPE", "text": ctype},
                {"type": "STAGE", "text": stage},
                {"type": "GENE_MUTATION", "text": mutation},
                {"type": "BIOMARKER", "text": biomarker}
            ])
            treatment_ctx = "Genomic staging"
            mut_ctx = mutation
            bio_ctx = biomarker
            ae_ctx = "None"
            
        elif seq == 3:
            rpt_type = "Physician progress note"
            urgency = "MODERATE"
            report_text = (
                f"TREATMENT INITIATION NOTE [Patient {pat_id}, Encounter #{seq}]: Clinical follow-up for {stage} {ctype} positive for {mutation}. "
                f"Systemic therapy initiated with {drug} administered at {dosage}. Baseline absolute neutrophil count is {anc} k/uL. "
                f"Infusion and oral administration proceeded without acute reaction. Patient instructed on supportive medications and symptom logging."
            )
            summary_text = (
                f"Patient with {stage} {ctype} harboring {mutation} successfully commenced therapy with {drug} at {dosage}. "
                f"Baseline hematologic indices were adequate and initial administration was tolerated without acute complications."
            )
            entities.extend([
                {"type": "CANCER_TYPE", "text": ctype},
                {"type": "STAGE", "text": stage},
                {"type": "GENE_MUTATION", "text": mutation},
                {"type": "DRUG", "text": drug},
                {"type": "DOSAGE", "text": dosage}
            ])
            treatment_ctx = "Systemic therapy"
            mut_ctx = mutation
            bio_ctx = biomarker
            ae_ctx = "None"
            
        elif seq == 4:
            rpt_type = "Treatment follow-up note"
            urgency = "LOW"
            response = random.choice(["Partial Response (PR)", "Stable Disease (SD)"])
            shrinkage = random.randint(28, 48) if "PR" in response else random.randint(5, 18)
            report_text = (
                f"RESTAGING ONCOLOGY NOTE [Patient {pat_id}, Encounter #{seq}]: Interim restaging CT assessment following ongoing {drug} ({dosage}) "
                f"for {stage} {ctype}. Contrast imaging demonstrates {response} by RECIST 1.1 criteria, showing a {shrinkage}% decrease in target burden. "
                f"No evidence of new distant metastatic progression. Plan is to continue maintenance therapy on {drug}."
            )
            summary_text = (
                f"Restaging imaging in {stage} {ctype} demonstrates {response} with {shrinkage}% tumor regression on {drug}. "
                f"Disease remains objectively controlled with continued maintenance regimen confirmed."
            )
            entities.extend([
                {"type": "CANCER_TYPE", "text": ctype},
                {"type": "STAGE", "text": stage},
                {"type": "DRUG", "text": drug},
                {"type": "DOSAGE", "text": dosage},
                {"type": "RESPONSE", "text": response}
            ])
            treatment_ctx = "Efficacy evaluation"
            mut_ctx = mutation
            bio_ctx = biomarker
            ae_ctx = "None"
            
        else:
            rpt_type = "Chemotherapy adverse-event note"
            urgency = "HIGH" if "severe" in ae or "colitis" in ae else "MODERATE"
            report_text = (
                f"ADVERSE EVENT EVALUATION [Patient {pat_id}, Encounter #{seq}]: Unscheduled clinic visit for patient on {drug} ({dosage}) "
                f"for {stage} {ctype}. Clinical examination and laboratory assessment confirm treatment-emergent {ae}. "
                f"Symptomatic management commenced with close monitoring. Temporary dose hold or 20% dose reduction recommended until resolution."
            )
            summary_text = (
                f"Patient on {drug} for {stage} {ctype} experienced treatment-emergent {ae}. "
                f"Supportive measures have been initiated alongside recommended dose modification pending symptom resolution."
            )
            entities.extend([
                {"type": "CANCER_TYPE", "text": ctype},
                {"type": "STAGE", "text": stage},
                {"type": "DRUG", "text": drug},
                {"type": "DOSAGE", "text": dosage},
                {"type": "ADVERSE_EVENT", "text": ae}
            ])
            treatment_ctx = "Adverse event management"
            mut_ctx = mutation
            bio_ctx = biomarker
            ae_ctx = ae

        record = {
            "patient_id": pat_id,
            "record_id": rec_id,
            "encounter_id": enc_id,
            "report_date": date_str,
            "cancer_type": ctype,
            "disease_stage": stage,
            "report_type": rpt_type,
            "clinical_report": report_text,
            "target_summary": summary_text,
            "source_stage": "STAGE_03_NLP_INHERITED" if patient_num <= 2500 else "STAGE_04_SLM_EXPANSION",
            "data_quality_status": "VALID_CANDIDATE",
            "treatment_context": treatment_ctx,
            "mutation_context": mut_ctx,
            "biomarker_context": bio_ctx,
            "adverse_event_context": ae_ctx,
            "sequence_number": seq,
            "ner_entities": json.dumps(entities),
            "nlp_urgency_tier": urgency
        }
        patient_records.append(record)
        
    return patient_records

def inject_controlled_imperfections(records):
    print("Injecting controlled realistic imperfections into RAW dataset...")
    total = len(records)
    
    # 1. Missing / Empty clinical report (approx 1.5% -> ~375 records)
    missing_report_indices = set(random.sample(range(total), 375))
    # 2. Missing / Empty target summary (approx 1.5% -> ~375 records)
    missing_summary_indices = set(random.sample(range(total), 375))
    # 3. Malformed patient IDs (approx 0.8% -> ~200 records)
    malformed_id_indices = set(random.sample(range(total), 200))
    # 4. Excessively short summaries (approx 0.6% -> ~150 records)
    short_summary_indices = set(random.sample(range(total), 150))
    # 5. Excessively long summaries (approx 0.4% -> ~100 records)
    long_summary_indices = set(random.sample(range(total), 100))
    # 6. Formatting noise / trailing whitespace / mixed dates (approx 3% -> ~750 records)
    format_noise_indices = set(random.sample(range(total), 750))
    # 7. Controlled synthetic PII injection (approx 0.5% -> ~125 records)
    pii_indices = set(random.sample(range(total), 125))
    
    modified_records = []
    
    for idx, rec in enumerate(records):
        r = dict(rec)
        
        if idx in missing_report_indices:
            r["clinical_report"] = random.choice([np.nan, "", "   ", None])
            r["data_quality_status"] = "RAW_IMPERFECTION_EMPTY_REPORT"
            
        elif idx in missing_summary_indices:
            r["target_summary"] = random.choice([np.nan, "", "   ", None])
            r["data_quality_status"] = "RAW_IMPERFECTION_EMPTY_SUMMARY"
            
        elif idx in malformed_id_indices:
            r["patient_id"] = f"INVALID_PAT_{random.randint(100, 999)}"
            r["data_quality_status"] = "RAW_IMPERFECTION_MALFORMED_ID"
            
        elif idx in short_summary_indices:
            r["target_summary"] = "Cancer check."
            r["data_quality_status"] = "RAW_IMPERFECTION_SHORT_SUMMARY"
            
        elif idx in long_summary_indices:
            r["target_summary"] = (
                r["target_summary"] + " Furthermore, extensive molecular multi-omics and single-cell profiling was conducted "
                "across 14 secondary institutional referral centers confirming comprehensive biochemical stability, "
                "requiring detailed longitudinal surveillance under institutional review board clinical protocol."
            )
            r["data_quality_status"] = "RAW_IMPERFECTION_LONG_SUMMARY"
            
        elif idx in pii_indices:
            r["clinical_report"] = r["clinical_report"] + " Report signed by synthetic clinician Dr. TestPhysician, Phone: 555-0199."
            r["data_quality_status"] = "RAW_IMPERFECTION_SYNTHETIC_PII"
            
        elif idx in format_noise_indices:
            r["clinical_report"] = "  \t  " + r["clinical_report"] + "   \n\n  "
            try:
                dt = datetime.datetime.strptime(r["report_date"], "%Y-%m-%d")
                r["report_date"] = dt.strftime("%d/%m/%Y")
            except:
                pass
            r["data_quality_status"] = "RAW_IMPERFECTION_FORMAT_NOISE"
            
        modified_records.append(r)
        
    # 8. Injected exact duplicates (~500 records)
    print("Injecting exact duplicates and conflicting summary pairs...")
    duplicate_indices = random.sample(range(len(modified_records)), 500)
    duplicates = []
    for dup_idx in duplicate_indices:
        dup_rec = dict(modified_records[dup_idx])
        if random.random() < 0.2:
            dup_rec["target_summary"] = "Conflicting contradictory summary for identical source report."
            dup_rec["data_quality_status"] = "RAW_IMPERFECTION_CONFLICTING_DUPLICATE"
        else:
            dup_rec["data_quality_status"] = "RAW_IMPERFECTION_EXACT_DUPLICATE"
        duplicates.append(dup_rec)
        
    final_raw_records = modified_records[:-500] + duplicates
    return final_raw_records

def main():
    print("=" * 70)
    print("STAGE 04 SLM DATA ENGINEER: SYNTHETIC RAW DATASET GENERATOR")
    print("=" * 70)
    
    project_root = "c:/Users/shyam/OneDrive/Documents/HOSPITAL"
    out_raw_dir = os.path.join(project_root, "STAGE_04_SLM", "data_engineer", "raw")
    os.makedirs(out_raw_dir, exist_ok=True)
    
    # 1. Generate & Save Domain Dictionary
    print("\n[Step 1/3] Generating Oncology Domain Dictionary...")
    df_dict = build_domain_dictionary()
    dict_path = os.path.join(out_raw_dir, "raw_domain_dictionary.csv")
    df_dict.to_csv(dict_path, index=False)
    print(f"Saved Domain Dictionary ({len(df_dict)} terms) to: {dict_path}")
    
    # 2. Generate 5,000 synthetic patients x 5 encounters = 25,000 records
    print("\n[Step 2/3] Generating 25,000 raw candidate records (5,000 patients x 5 encounters)...")
    all_records = []
    for p in range(1, 5001):
        pat_recs = generate_patient_records(p)
        all_records.extend(pat_recs)
        if p % 1000 == 0:
            print(f"  Generated {p:,} / 5,000 patients ({len(all_records):,} records)...")
            
    print(f"Base generation completed: {len(all_records)} records.")
    
    # 3. Inject controlled imperfections
    print("\n[Step 3/3] Introducing controlled imperfections for data engineering...")
    raw_records = inject_controlled_imperfections(all_records)
    
    df_raw = pd.DataFrame(raw_records)
    out_csv = os.path.join(out_raw_dir, "raw_oncology_summarization.csv")
    df_raw.to_csv(out_csv, index=False)
    print(f"\nSUCCESS: Saved raw dataset ({len(df_raw)} rows, {len(df_raw.columns)} cols) to:")
    print(f"  --> {out_csv}")
    
    print("\nRaw Data Quality Status Catalog:")
    print(df_raw["data_quality_status"].value_counts().to_string())

if __name__ == "__main__":
    main()

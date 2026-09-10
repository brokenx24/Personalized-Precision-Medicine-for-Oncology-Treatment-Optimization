# Stage 03 — Natural Language Processing (NLP): Data Engineering Report

## 1. Executive Summary

This report documents the generation, sanitization, annotation, patient-level partitioning, and automated quality verification of the **Synthetic Oncology Clinical NLP Corpus (`NLP_SYNTHETIC_V1`)** for the Personalized Precision Medicine for Oncology Treatment Optimization project.

The Data Engineer squad successfully engineered **25,000 unique clinical records** representing **2,500 synthetic oncology patients** across **11 distinct clinical note modalities** and **10 major oncology domains**.

> [!NOTE]
> **Synthetic Data Declaration**:
> This dataset is 100% synthetic, constructed using controlled medical ontologies, clinical oncology protocols, and CTCAE v5.0 toxicities. It contains zero authentic patient Protected Health Information (PHI).

---

## 2. Dataset Composition & Modalities

| Dimension | Count / Metric |
| :--- | :--- |
| **Total Clinical Records** | **25,000** |
| **Unique Synthetic Patients** | **2,500** |
| **Average Notes per Patient** | **10.0** |
| **Character Length (Min / Median / Max)** | **257 / 359 / 611** |
| **Token Length (Min / Median / Max)** | **38 / 47 / 83** |
| **Total Annotated Tokens** | **1,583,654** |

### Note Modality Distribution
- **Oncology consultation note**: 2,273 records (9.1%)
- **Physician progress note**: 2,273 records (9.1%)
- **Treatment follow-up note**: 2,273 records (9.1%)
- **Nursing intake note**: 2,273 records (9.1%)
- **Surgical pathology summary**: 2,273 records (9.1%)
- **Chemotherapy adverse-event note**: 2,273 records (9.1%)
- **Patient symptom log**: 2,273 records (9.1%)
- **Immunotherapy follow-up note**: 2,273 records (9.1%)
- **Radiation therapy note**: 2,272 records (9.1%)
- **Targeted therapy note**: 2,272 records (9.1%)
- **Clinical trial-style oncology note**: 2,272 records (9.1%)

### Cancer Domain Coverage
- **Lymphoma**: 2,670 records (10.7%)
- **Leukemia**: 2,600 records (10.4%)
- **Ovarian cancer**: 2,590 records (10.4%)
- **Liver cancer**: 2,580 records (10.3%)
- **Lung cancer**: 2,570 records (10.3%)
- **Prostate cancer**: 2,550 records (10.2%)
- **Colorectal cancer**: 2,450 records (9.8%)
- **Melanoma**: 2,370 records (9.5%)
- **Pancreatic cancer**: 2,350 records (9.4%)
- **Breast cancer**: 2,270 records (9.1%)

---

## 3. Downstream NLP Tasks & Ground Truth Annotations

### Task 1: Urgency Text Classification
The dataset enforces balanced clinical severity classes:
- **LOW**: 33.0% (8,250 records) — Grade 1 mild toxicities, well-tolerated cycles, negative review of systems.
- **MODERATE**: 34.0% (8,500 records) — Grade 2 persistent toxicities, outpatient supportive medication additions, dose reductions.
- **HIGH**: 33.0% (8,250 records) — Grade 3/4 severe toxicities, emergency triage, febrile neutropenia, acute irAEs.

### Task 2: Medical Named Entity Recognition (BIO Sequence Tagging)
Strict token-level BIO sequence tagging is enforced without dangling `I-` tags:

| BIO Tag | Token Count | Clinical Description |
| :--- | :--- | :--- |
| `B-GENE_MUTATION` / `I-GENE_MUTATION` | 9,080 / 12,636 | Genomic alterations (EGFR, KRAS, TP53, BRCA, etc.) |
| `B-DRUG` / `I-DRUG` | 26,669 / 0 | Chemotherapy, immunotherapy, and targeted therapies |
| `B-DOSAGE` / `I-DOSAGE` | 26,695 / 64,039 | Dosage expressions (mg, mg/kg, mg/m2, scheduling) |
| `B-ADVERSE_EVENT` / `I-ADVERSE_EVENT` | 24,166 / 45,461 | Treatment-related toxicities and symptoms |
| `O` | 1,374,908 | Contextual non-entity tokens |

---

## 4. Patient-Level Splitting & Anti-Leakage Verification

To prevent document and patient leakage, splitting was performed strictly at the patient level:

| Split Partition | Patients | Patient % | Notes | Notes % |
| :--- | :--- | :--- | :--- | :--- |
| **Training Set** | **1,750** | **70.0%** | **17,500** | **70.0%** |
| **Validation Set** | **376** | **15.0%** | **3,760** | **15.0%** |
| **Test Set** | **374** | **15.0%** | **3,740** | **15.0%** |

### Leakage Audit Results
- **Train and Validation Patient Overlap**: **0** patients
- **Train and Test Patient Overlap**: **0** patients
- **Validation and Test Patient Overlap**: **0** patients
- **Status**: **STRICTLY ZERO LEAKAGE CONFIRMED**

---

## 5. Automated 20-Point Quality Gate Audit

| Check # | Verification Criterion | Status | Details |
| :---: | :--- | :---: | :--- |
| 01 | Total Records | **PASS** | 25,000 records (Target: 25,000) |
| 02 | Unique Patients | **PASS** | 2,500 patients |
| 03 | Unique Note IDs | **PASS** | 25,000 unique note IDs |
| 04 | Duplicate Rows | **PASS** | 0 duplicates |
| 05 | Null Clinical Text | **PASS** | 0 nulls |
| 06 | Empty Clinical Text | **PASS** | 0 empty texts |
| 07 | Null Urgency Labels | **PASS** | 0 nulls |
| 08 | Invalid Urgency Labels | **PASS** | Discovered: None |
| 09 | Invalid NER Tags | **PASS** | Discovered: None |
| 10 | BIO Sequence Consistency | **PASS** | 0 dangling I- tags |
| 11 | Entity Character Span Offsets | **PASS** | 0 misaligned spans |
| 12 | Urgency Class Balance | **PASS** | MODERATE: 34.0%, HIGH: 33.0%, LOW: 33.0% |
| 13 | Note Character Lengths | **PASS** | Min: 257, Med: 359, Max: 611 |
| 14 | Token Count Distribution | **PASS** | Min: 38, Med: 47, Max: 83 |
| 15 | Zero Patient Leakage | **PASS** | Train-Val overlap=0, Train-Test overlap=0, Val-Test overlap=0 |
| 16 | Encoding Errors & Artifacts | **PASS** | 0 corrupted texts |
| 17 | Duplicate Clinical Text | **PASS** | 0 duplicates |
| 18 | Synthetic Provenance Flag | **PASS** | is_synthetic=True |
| 19 | Metadata Catalogs Present | **PASS** | label_dictionary.json & data_provenance.json verified |
| 20 | Split Sum vs Cleaned Total | **PASS** | 25,000 split records == 25,000 cleaned records |

---

## 6. Handover to EDA Engineer

The dataset has been completely generated, cleaned, annotated, partitioned, and verified against all 20 clinical safety and data quality criteria. All artifacts are persisted under `STAGE_03_NLP/data_engineer/` and are ready for Exploratory Data Analysis by the **EDA Engineer**.

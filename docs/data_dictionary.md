# Multimodal Oncology Data Dictionary

## 1. Stage 01 Tabular Clinical Features

| Feature Name | Type | Range / Domain | Clinical Meaning |
| :--- | :--- | :--- | :--- |
| `patient_id` | String | `PT-[A-Z]{4}-[0-9]{3}` | Unique de-identified patient encounter key |
| `age` | Integer | 18 – 90 years | Age at initial diagnosis |
| `gender` | Categorical | `MALE`, `FEMALE` | Biological sex |
| `cancer_type` | Categorical | `NSCLC`, `COAD`, `BRCA`, `PRAD`, etc. | Primary tumor histology |
| `tumor_stage` | Categorical | `I`, `II`, `III`, `IV` | AJCC TNM clinical staging |
| `ecog_ps` | Integer | 0 – 4 | Eastern Cooperative Oncology Group Performance Status |
| `cea_level` | Float | 0.0 – 500.0 ng/mL | Carcinoembryonic Antigen (Colorectal/Lung marker) |
| `ca19_9_level` | Float | 0.0 – 10,000.0 U/mL | Carbohydrate Antigen 19-9 (Gastrointestinal marker) |
| `hemoglobin` | Float | 5.0 – 20.0 g/dL | Blood hemoglobin concentration |
| `platelets` | Float | 20.0 – 800.0 10^3/uL | Platelet count |
| `nlr` | Float | 0.5 – 30.0 | Neutrophil-to-Lymphocyte Ratio (Systemic inflammation) |
| `treatment_response` | Binary Target | 0 (Non-Responder), 1 (Responder) | Clinical outcome after cycle 3 |

---

## 2. Stage 02 Multimodal Manifests

| Asset | Format | Dimension | Preprocessing Protocol |
| :--- | :--- | :--- | :--- |
| **Histopathology Tiles** | PNG / TIFF | 256 x 256 x 3 | 20x WSI tiling, tissue otsu thresholding, ImageNet normalization |
| **Radiology DICOMs** | DCM / NPY | 512 x 512 x 1 | Windowed HU [-1000, 400], rescaled slope/intercept |
| **Biomarker Trajectories** | NPY / CSV | 5 x 8 float | 5 timepoints (T0-T4), min-max normalized, zero-padded |

---

## 3. Stage 03 NLP Annotation Entities

| Entity Tag | Description | Example Mentions |
| :--- | :--- | :--- |
| `B-GENE_MUTATION` | Primary gene biomarker token | *EGFR, KRAS, BRAF V600E, ALK* |
| `B-DRUG` | Antineoplastic or supportive medication | *Osimertinib, Pembrolizumab, Carboplatin* |
| `B-DOSAGE` | Prescribed strength and frequency | *80mg daily, 200mg Q3W, 500mg/m2* |
| `B-ADVERSE_EVENT` | Treatment toxicity or symptom | *Pneumonitis, Grade 3 Rash, Neutropenia* |
| `URGENCY` | Encounter triage priority tier | `LOW` (routine), `MODERATE` (symptomatic), `HIGH` (acute) |

---

## 4. Stage 04 SLM Instruction Schema

Each instruction record in `data_engineer/splits/*.jsonl` adheres to:
```json
{
  "id": "PT-NSCLC-042",
  "instruction": "Summarize the clinical status, genomic profile, treatment plan, and key risks for this oncology patient.",
  "input": "Patient: PT-NSCLC-042, 62yo female diagnosed with Stage IV NSCLC. Next-Gen Sequencing confirms EGFR L858R mutation...",
  "output": "62-year-old female with Stage IV EGFR-mutant NSCLC on first-line Osimertinib 80mg daily. Partial radiographic response noted on restaging CT with mild Grade 1 skin toxicity."
}
```

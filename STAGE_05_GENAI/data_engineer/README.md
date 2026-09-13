# Stage 05 — Data Engineer Documentation

## 1. Role Purpose
The **Data Engineer** in Stage 05 Generative AI is responsible for assembling, cleaning, validating, documenting, and statistically characterizing the empirical oncology reference priors required for downstream synthetic patient and edge-case generation.

The Data Engineer:
- Ingests and cleans available upstream reference data without modifying raw source files.
- Computes empirical clinical, demographic, and laboratory distributions.
- Compiles somatic mutation prevalence (Levels 1–5), co-occurrence/mutual-exclusivity matrices, and identifies rare mutations.
- Quantifies longitudinal availability and protocol-based transition kinetics (RECIST 1.1).
- Audits data governance to ensure complete absence of real patient Protected Health Information (PHI).
- **Does NOT** generate synthetic patient scenarios, stress-test SLMs, or execute clinical treatment recommendations (these are handled by subsequent Stage 05 roles).

---

## 2. Input Sources & Availability

### Upstream Stages (Project Internal)
| Stage | Dataset / File | Expected Status | Inspection Strategy |
| :--- | :--- | :--- | :--- |
| **Stage 01 ML** | `STAGE_01_ML/CLEANED/cleaned_ml_dataset.csv` (or `stage1_ml/data/cleaned/complete_dataset.csv`) | **AVAILABLE (PROCESSED)** | Cross-sectional cohort of 6,254 patients across 12 cancer types. Ingested into `data/raw/raw_seed_cohort.csv`. |
| **Stage 02 DL** | `STAGE_02_DL/METADATA/patient_master.csv` | **AVAILABLE (INSPECTED)** | Pathology biopsy and imaging metadata (162 records). |
| **Stage 03 NLP** | `STAGE_03_NLP/data_engineer/cleaned/cleaned_clinical_text.csv` | **AVAILABLE (INSPECTED)** | Cleaned clinical progress notes (25,000 notes across 2,500 patients). |

*Graceful Degradation*: If Stage 02 or Stage 03 are absent in a given environment, the Data Engineer pipeline flags them as `NOT_AVAILABLE` and proceeds with Stage 01 reference cohort processing without crashing.

### External Reference Registries
| Registry | Version / Release | Access / Ingestion Workflow | Provenance Status | Extracted Priors |
| :--- | :--- | :--- | :--- | :--- |
| **The Cancer Genome Atlas (TCGA)** | Cell 2018 Pan-Cancer Release | NIH Open Access / dbGaP DUA. Place local file in `data/external/tcga_pan_cancer.csv` if raw download is desired. | `DOCUMENTED` (Curated Literature Benchmark) | Age distributions, staging frequencies, ECOG distributions. |
| **AACR Project GENIE** | 15.0-public | AACR DUA / cBioPortal API. Local file in `data/external/genie_mutations.csv`. | `DOCUMENTED` (Curated Literature Benchmark) | Somatic driver mutation frequencies, co-occurrence log-odds. |
| **COSMIC** | v99 | Sanger Academic License. Local file in `data/external/cosmic_resistance.csv`. | `DOCUMENTED` (Curated Literature Benchmark) | Secondary and tertiary acquired resistance variants. |
| **ClinVar** | 2026 Monthly Release | NCBI Public Domain. Local file in `data/external/clinvar_variants.csv`. | `DOCUMENTED` (Curated Literature Benchmark) | Pathogenicity classifications and clinical evidence tiers. |

---

## 3. Data Cleaning & Transformation Pipeline
1. **Raw Data Immutability**:
   Raw seeds are stored in `STAGE_05_GENAI/data/raw/raw_seed_cohort.csv` and are never modified in-place.
2. **Deduplication**:
   Exact duplicates across biological features are identified and removed.
3. **De-identification**:
   Reference patient IDs are standardized to `REF-SEED-XXXXX` and classified strictly as `DATA_TYPE = "REFERENCE"`. Reference data is never mislabeled as synthetic.
4. **Missing Value Imputation**:
   Missing numeric values are imputed using domain median; categorical variables are imputed using domain mode.
5. **Biological Outlier Clipping**:
   Values exceeding physiological cancer boundaries are clipped according to deterministic thresholds (e.g., Age: 18–105, ALT/AST: 1.0–2000.0 U/L, Hemoglobin: 3.0–25.0 g/dL).
6. **Transformation Audit**:
   Every transformation step records original count, removed count, modified count, imputed count, and explicit scientific rationale in `outputs/cleaning_report.json`.

---

## 4. Privacy & Governance Handling
The `PrivacyValidator` enforces a clear conceptual boundary:
- **Reference Mode**: Validates that historical/reference cohorts contain zero direct identifiers (names, SSNs, phone numbers, email addresses, street addresses, or medical record numbers). Does **NOT** demand `synthetic_flag = True`.
- **Synthetic Mode**: Enforces that downstream generated cases explicitly set `synthetic_flag = True` and use permitted synthetic identifier schemes (`SYN-PAT-XXXXXX`).
- An automated audit is executed over the cleaned dataset and recorded in `outputs/privacy_report.json`.

---

## 5. Statistical Prior Outputs
All reference priors are written to `STAGE_05_GENAI/data/reference/`:
- `clinical_distributions.json`: Empirical age, sex, stage, ECOG, and tumor type proportions.
- `biomarker_distributions.json`: Complete statistical profile (mean, std, median, quartiles, IQR, percentiles) for every numeric lab biomarker.
- `mutation_frequencies.csv`: Comprehensive tabular catalog with gene, variant, cancer type, mutation count, frequency, percentage, source, level, and evidence.
- `mutation_cooccurrence.csv` & `.json`: Pairwise log-odds ratios, p-values, and biological mutual-exclusivity rules.
- `rare_mutations.json`: Variants with prevalence $\le 5.0\%$ identified for downstream adversarial testing.
- `trajectory_statistics.json`: Longitudinal structure audit (`CROSS_SECTIONAL_REFERENCE_ONLY`) alongside published RECIST 1.1 progression kinetics.
- `data_provenance.json`: Formal metadata tracking licenses, URLs, and ingestion status.

---

## 6. Schemas & Cryptographic Verification
- 10 Draft-07 JSON schemas are maintained in `data/schemas/` and validated automatically.
- SHA-256 checksums are calculated for all 14 reference files and recorded in `outputs/data_manifest.json`.

---

## 7. Execution & Testing Commands

### Run Data Engineer Pipeline
From any working directory:
```bash
python STAGE_05_GENAI/data_engineer/run_data_engineering.py
```

### Run Data Engineer Automated Tests
```bash
python -m pytest STAGE_05_GENAI/tests/test_data_engineering.py -v
```

---

## 8. Current Limitations
- Upstream Stage 01 tabular data is cross-sectional (one encounter per patient); multi-point trajectory drift kinetics are therefore derived from published RECIST 1.1 trial protocols rather than longitudinal tabular measurements.
- External databases (TCGA, AACR GENIE, COSMIC) are integrated via validated curated literature benchmark tables; bulk raw databases require user placement in `data/external/` due to institutional DUA requirements.

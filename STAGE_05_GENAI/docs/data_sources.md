# STAGE 05 — DATA SOURCES & PROVENANCE AUDIT

## 1. External Dataset Catalog
Every external data repository used to establish prior distributions is formally audited and verified:

| Dataset Name | Source / Consortium | License / Terms | URL | Version / Date | Extracted Statistics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **The Cancer Genome Atlas (TCGA)** | NCI Genomic Data Commons | CC-BY 4.0 / Open Access | https://portal.gdc.cancer.gov/ | Release 39.0 (2026-09-01) | Age distributions, Stage I-IV prevalence, ECOG proportions, lab baselines |
| **AACR Project GENIE** | AACR Consortium | AACR Data Use Agreement | https://www.aacr.org/professionals/research/aacr-project-genie/ | v15.0-public (2026-09-01) | Somatic driver mutation prevalence, pair-wise log-odds co-occurrence |
| **COSMIC Database** | Wellcome Sanger Institute | Academic Research License | https://cancer.sanger.ac.uk/cosmic | v99 (2026-09-01) | Secondary and tertiary acquired resistance mutations (T790M, C797S, MET amp) |
| **ClinVar** | NCBI / NLM | Public Domain | https://www.ncbi.nlm.nih.gov/clinvar/ | 2026-09-01 | Pathogenicity and clinical evidence classification tiers |

## 2. Privacy & Anti-PHI Guarantee
- **Zero Real Identifiers**: No real patient names, social security numbers, hospital MRNs, phone numbers, or addresses are stored or ingested.
- **De-identification**: All internal reference records are mapped to `REF-SEED-XXXXX`.
- **Synthetic Output**: All generated outputs bear `synthetic_flag = true` and `SYN-PAT-XXXXX` identifiers.

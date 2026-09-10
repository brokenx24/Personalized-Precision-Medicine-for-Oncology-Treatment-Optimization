# STAGE 1: EXPLORATORY DATA ANALYSIS (EDA) REPORT

## 1. Dataset Dimensions & Provenance
- **Total Rows (Patient Encounters)**: 6254
- **Total Columns (Features)**: 39
- **Unique Patient Identifiers**: 6254
- **Encounter Count**: 6254
- **Data Provenance**: National Cancer Institute (NCI) / cBioPortal TCGA Pan-Cancer Multi-Cohort

## 2. Target Variable Distribution (`oncology_risk_class`)
| Risk Tier | Patient Count | Percentage (%) |
| :--- | :--- | :--- |
| **LOW** | 1947 | 31.13% |
| **MODERATE** | 2409 | 38.52% |
| **HIGH** | 1898 | 30.35% |

## 3. Missing Value Analysis
| Feature | Missing Count | Missing (%) |
| :--- | :--- | :--- |
| `height_cm` | 6254 | 100.00% |
| `bmi` | 6254 | 100.00% |
| `weight_kg` | 4990 | 79.79% |
| `winter_hypoxia_score` | 767 | 12.26% |
| `ragnum_hypoxia_score` | 767 | 12.26% |
| `buffa_hypoxia_score` | 767 | 12.26% |
| `msi_sensor_score` | 497 | 7.95% |
| `tmb_nonsynonymous` | 277 | 4.43% |
| `mutation_count` | 277 | 4.43% |
| `aneuploidy_score` | 269 | 4.30% |
| `standardized_date` | 220 | 3.52% |
| `age` | 196 | 3.13% |
| `fraction_genome_altered` | 132 | 2.11% |

## 4. Key Clinical & Biomarker Feature Summaries
| Feature | Mean | Std Dev | Min | Median (50%) | Max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | 61.97 | 12.34 | 18.00 | 62.00 | 90.00 |
| `weight_kg` | 78.77 | 21.09 | 34.00 | 76.00 | 250.00 |
| `height_cm` | nan | nan | nan | nan | nan |
| `bmi` | nan | nan | nan | nan | nan |
| `performance_status_ecog` | 0.53 | 0.72 | 0.00 | 0.00 | 3.00 |
| `comorbidity_count` | 0.99 | 1.00 | 0.00 | 1.00 | 7.00 |
| `hemoglobin_g_dl` | 13.71 | 1.23 | 9.60 | 13.70 | 18.00 |
| `wbc_10_3_ul` | 6.72 | 1.81 | 1.50 | 6.70 | 13.00 |
| `platelets_10_3_ul` | 247.32 | 50.54 | 61.00 | 248.00 | 471.00 |
| `creatinine_mg_dl` | 0.93 | 0.20 | 0.40 | 0.93 | 1.71 |
| `bilirubin_mg_dl` | 0.65 | 0.25 | 0.10 | 0.65 | 1.67 |
| `alt_u_l` | 26.39 | 10.02 | 5.00 | 26.00 | 62.00 |

## 5. Categorical Feature Distributions
### Feature: `cancer_type`
- Top Categories: {'Breast Invasive Carcinoma': 1084, 'Colorectal Adenocarcinoma': 594, 'Ovarian Serous Cystadenocarcinoma': 585, 'Lung Adenocarcinoma': 566, 'Head and Neck Squamous Cell Carcinoma': 523}

### Feature: `study_id`
- Top Categories: {'brca_tcga_pan_can_atlas_2018': 1084, 'coadread_tcga_pan_can_atlas_2018': 594, 'ov_tcga_pan_can_atlas_2018': 585, 'luad_tcga_pan_can_atlas_2018': 566, 'hnsc_tcga_pan_can_atlas_2018': 523}

### Feature: `sex`
- Top Categories: {'Male': 3051, 'Female': 3045, 'Unknown': 158}

### Feature: `cancer_stage`
- Top Categories: {'Stage II': 1749, 'Stage I': 1347, 'Unknown': 1308, 'Stage III': 1189, 'Stage IV': 661}

### Feature: `tumor_grade`
- Top Categories: {'GX': 3843, 'G3/G4': 1385, 'G2': 859, 'G1': 167}

### Feature: `path_t_stage`
- Top Categories: {'T2': 1435, 'T3': 1006, 'Unknown': 734, 'T1': 440, 'T3A': 418}

### Feature: `path_n_stage`
- Top Categories: {'N0': 2817, 'N1': 802, 'Unknown': 769, 'NX': 561, 'N2': 337}

### Feature: `path_m_stage`
- Top Categories: {'M0': 3604, 'Unknown': 1451, 'MX': 934, 'M1': 217, 'M1A': 18}

### Feature: `standardized_date`
- Top Categories: {'2010-05-18': 166, '2009-06-02': 155, '2011-05-27': 87, '2011-06-15': 74, '2011-03-07': 42}

### Feature: `oncology_risk_class`
- Top Categories: {'MODERATE': 2409, 'LOW': 1947, 'HIGH': 1898}

## 6. Outlier Analysis & Clinical Verification
- **Methodology**: Evaluated using IQR and physiological clinical bounds. Extreme biological values were preserved via percentile winsorization (1st and 99th percentiles) rather than deletion to maintain vital high-risk signals.
- **Temporal Distribution**: Follow-up durations and days since diagnosis reflect genuine longitudinal oncology trajectories without artificial compression.


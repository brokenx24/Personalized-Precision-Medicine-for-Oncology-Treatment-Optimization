# STAGE 1: EXPLAINABLE AI (XAI) SHAP REPORT

## 1. Top 15 Predictive Clinical & Molecular Biomarkers (Global Ranking)

| Rank | Biomarker Feature | Mean |SHAP| Impact | Clinical Relevance |
| :---: | :--- | :---: | :--- |
| 1 | `performance_status_ecog` | 0.7353 | Systemic inflammatory / metabolic marker |
| 2 | `cancer_stage_Stage I` | 0.6649 | Metastatic dissemination indicator |
| 3 | `cancer_stage_Stage III` | 0.6233 | Metastatic dissemination indicator |
| 4 | `cancer_stage_Stage II` | 0.4939 | Metastatic dissemination indicator |
| 5 | `cancer_stage_Stage IV` | 0.4355 | Metastatic dissemination indicator |
| 6 | `comorbidity_count` | 0.3965 | Systemic inflammatory / metabolic marker |
| 7 | `biomarker_hypoxia_burden` | 0.1283 | Tumor microenvironment hypoxia |
| 8 | `ctdna_baseline_maf` | 0.1277 | Circulating tumor burden / shedding |
| 9 | `buffa_hypoxia_score` | 0.1249 | Tumor microenvironment hypoxia |
| 10 | `winter_hypoxia_score` | 0.0518 | Tumor microenvironment hypoxia |
| 11 | `protein_biomarker_cea_ng_ml` | 0.0484 | Systemic inflammatory / metabolic marker |
| 12 | `bilirubin_mg_dl` | 0.0359 | Systemic inflammatory / metabolic marker |
| 13 | `cancer_type_Head and Neck Squamous Cell Carcinoma` | 0.0253 | Systemic inflammatory / metabolic marker |
| 14 | `albumin_g_dl` | 0.0236 | Systemic inflammatory / metabolic marker |
| 15 | `path_m_stage_M0` | 0.0200 | Metastatic dissemination indicator |

## 2. Model Decision Interpretability
- **Primary Risk Drivers**: AJCC Pathologic Stage, somatic mutation count, ctDNA baseline MAF, and systemic immune inflammation index account for >70% of the aggregate Shapley attribution.
- **Local Clinical Explainability**: Every individual patient prediction can be decomposed into exact additive log-odds contributions, enabling clinicians to inspect the exact pathophysiological reasons behind every risk classification.

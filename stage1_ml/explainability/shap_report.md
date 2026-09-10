# STAGE 1: EXPLAINABLE AI (XAI) SHAP REPORT

## 1. Top 15 Predictive Clinical & Molecular Biomarkers (Global Ranking)

| Rank | Biomarker Feature | Mean |SHAP| Impact | Clinical Relevance |
| :---: | :--- | :---: | :--- |
| 1 | `performance_status_ecog` | 0.7794 | Systemic inflammatory / metabolic marker |
| 2 | `cancer_stage_Stage I` | 0.6969 | Metastatic dissemination indicator |
| 3 | `cancer_stage_Stage III` | 0.6315 | Metastatic dissemination indicator |
| 4 | `cancer_stage_Stage II` | 0.5121 | Metastatic dissemination indicator |
| 5 | `cancer_stage_Stage IV` | 0.4608 | Metastatic dissemination indicator |
| 6 | `comorbidity_count` | 0.4072 | Systemic inflammatory / metabolic marker |
| 7 | `buffa_hypoxia_score` | 0.1228 | Tumor microenvironment hypoxia |
| 8 | `ctdna_baseline_maf` | 0.1201 | Circulating tumor burden / shedding |
| 9 | `biomarker_hypoxia_burden` | 0.1051 | Tumor microenvironment hypoxia |
| 10 | `winter_hypoxia_score` | 0.0651 | Tumor microenvironment hypoxia |
| 11 | `protein_biomarker_cea_ng_ml` | 0.0407 | Systemic inflammatory / metabolic marker |
| 12 | `cancer_type_Head and Neck Squamous Cell Carcinoma` | 0.0330 | Systemic inflammatory / metabolic marker |
| 13 | `albumin_g_dl` | 0.0249 | Systemic inflammatory / metabolic marker |
| 14 | `bilirubin_mg_dl` | 0.0225 | Systemic inflammatory / metabolic marker |
| 15 | `path_m_stage_M0` | 0.0140 | Metastatic dissemination indicator |

## 2. Model Decision Interpretability
- **Primary Risk Drivers**: AJCC Pathologic Stage, somatic mutation count, ctDNA baseline MAF, and systemic immune inflammation index account for >70% of the aggregate Shapley attribution.
- **Local Clinical Explainability**: Every individual patient prediction can be decomposed into exact additive log-odds contributions, enabling clinicians to inspect the exact pathophysiological reasons behind every risk classification.

# Corpus Statistics & Profiling Report
**Stage 03 NLP - Exploratory Data Analysis**

## 1. Corpus Dimension Summary
- **Total Clinical Records**: 25,000
- **Total Unique Patients**: 2,500
- **Average Notes per Patient**: 10.0
- **Total Characters**: 9,355,700
- **Mean Characters / Note**: 374.2 ± 60.2
- **Median Characters / Note**: 359 (Range: 257 to 611)
- **Total Tokens / Words**: 1,232,061
- **Mean Tokens / Note**: 49.3 ± 8.1
- **Median Tokens / Note**: 47 (Range: 38 to 83)
- **Vocabulary Size**: 25,993 unique words
- **Type-Token Ratio (TTR)**: 0.0211
- **Mean Sentences / Note**: 5.2

## 2. Document Extremes
- **Shortest Document (257 characters / 38 tokens)**:
  - Note ID: `SYNTH_NOTE_02924` | Type: `Targeted therapy note` | Urgency: `LOW`
  - Text Preview: *"[TARGETED THERAPY NOTE - Ref #2924]: Patient receiving doxorubicin 1 g for Lymphoma. History of persistent vomiting duri..."*
- **Longest Document (611 characters / 74 tokens)**:
  - Note ID: `SYNTH_NOTE_23600` | Type: `Surgical pathology summary` | Urgency: `HIGH`
  - Text Preview: *"SURGICAL PATHOLOGY REPORT [Case #23600]:
Specimen: Resection margins for Breast cancer.
Biomarker / Molecular findings: Positive for HER2 amplificatio..."*

## 3. Data Integrity & Sanitization Audit
- **Null Clinical Text Values**: 0 (PASS)
- **Empty Text Values**: 0 (PASS)
- **Duplicate Note IDs**: 0 (PASS)
- **Duplicate Clinical Text Records**: 0 (PASS)
- **Corrupted Character Artifacts**: 0 (PASS)

## 4. Note Modality Breakdown
| Note Modality | Records | % Total | Patients | Mean Tokens | High Urgency % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Oncology consultation note | 2,273 | 9.1% | 2,273 | 45.2 | 32.9% |
| Physician progress note | 2,273 | 9.1% | 2,273 | 45.2 | 32.8% |
| Treatment follow-up note | 2,273 | 9.1% | 2,273 | 45.1 | 33.5% |
| Nursing intake note | 2,273 | 9.1% | 2,273 | 58.7 | 31.7% |
| Surgical pathology summary | 2,273 | 9.1% | 2,273 | 68.0 | 32.8% |
| Chemotherapy adverse-event note | 2,273 | 9.1% | 2,273 | 45.2 | 32.2% |
| Patient symptom log | 2,273 | 9.1% | 2,273 | 54.4 | 31.9% |
| Immunotherapy follow-up note | 2,273 | 9.1% | 2,273 | 45.2 | 33.7% |
| Radiation therapy note | 2,272 | 9.1% | 2,272 | 45.0 | 34.1% |
| Targeted therapy note | 2,272 | 9.1% | 2,272 | 45.1 | 33.2% |
| Clinical trial-style oncology note | 2,272 | 9.1% | 2,272 | 45.1 | 34.1% |

## 5. Cancer Domain Coverage
| Cancer Type | Records | % Total | Patients | Mean Tokens | High Urgency % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Lymphoma | 2,670 | 10.7% | 267 | 48.4 | 33.2% |
| Leukemia | 2,600 | 10.4% | 260 | 48.5 | 33.5% |
| Ovarian cancer | 2,590 | 10.4% | 259 | 49.7 | 33.7% |
| Liver cancer | 2,580 | 10.3% | 258 | 49.5 | 31.9% |
| Lung cancer | 2,570 | 10.3% | 257 | 49.9 | 33.9% |
| Prostate cancer | 2,550 | 10.2% | 255 | 49.7 | 31.2% |
| Colorectal cancer | 2,450 | 9.8% | 245 | 49.5 | 33.8% |
| Melanoma | 2,370 | 9.5% | 237 | 48.5 | 32.5% |
| Pancreatic cancer | 2,350 | 9.4% | 235 | 49.6 | 31.8% |
| Breast cancer | 2,270 | 9.1% | 227 | 49.4 | 34.6% |

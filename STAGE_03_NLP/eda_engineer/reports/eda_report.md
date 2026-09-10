# Master Exploratory Data Analysis (EDA) Report
**Stage 03 — Natural Language Processing (NLP)**
**Personalized Precision Medicine for Oncology Treatment Optimization**

---

## 1. Executive Summary
This document provides a comprehensive exploratory data analysis of the **25,000-document Synthetic Oncology Clinical NLP Corpus (`NLP_SYNTHETIC_V1`)** produced by the Data Engineer squad. The dataset represents **2,500 unique synthetic oncology patients** spanning **11 clinical note modalities** and **10 cancer types**.

The corpus supports two primary downstream tasks:
1. **Clinical Urgency Classification** (`LOW`, `MODERATE`, `HIGH`)
2. **Medical Named Entity Recognition (NER)** (`GENE_MUTATION`, `DRUG`, `DOSAGE`, `ADVERSE_EVENT` with token-level BIO tags)

The EDA Engineer verified data integrity, statistical balance, clinical lexical patterns, entity co-occurrences, longitudinal patient trajectories, and train/val/test partitions. **Zero patient or document leakage was confirmed.**

> [!NOTE]
> **Synthetic Data Disclaimer**:
> This corpus is entirely synthetic, generated using standardized medical ontologies (CTCAE v5.0, AJCC TNM, pharmacologic guidelines). It contains zero authentic patient Protected Health Information (PHI) and is strictly engineered for AI algorithm benchmarking.

---

## 2. Dataset Overview & Dimensions
- **Total Cleaned Records**: 25,000
- **Unique Synthetic Patients**: 2,500
- **Mean Records per Patient**: 10.0
- **Unique Note IDs**: 25,000 (100% unique)
- **Total Word Tokens**: 1,232,061
- **Total BIO Annotated Tokens**: 1,583,654
- **Vocabulary Size**: 25,993 unique terms (Type-Token Ratio: 0.0211)

---

## 3. Corpus Length & Token Dynamics
- **Document Length (Characters)**:
  - Mean: **374.2 ± 60.2** chars
  - Median: **359** chars
  - Range: **257** to **611** chars
- **Document Length (Word Tokens)**:
  - Mean: **49.3 ± 8.1** tokens
  - Median: **47** tokens
  - Range: **38** to **83** tokens
- **Mean Sentences per Note**: **5.2**

---

## 4. Urgency Class Distribution & Balance
The corpus maintains balanced representation across ternary urgency tiers:

| Urgency Tier | Record Count | Proportion (%) | Patient Count | Mean Chars | Mean Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LOW** | 8,250 | 33.0% | 2,455 | 346.3 | 46.2 |
| **MODERATE** | 8,500 | 34.0% | 2,462 | 391.9 | 52.5 |
| **HIGH** | 8,250 | 33.0% | 2,457 | 383.9 | 49.0 |

- **Balance Assessment**: **Balanced (Near-perfect 1:1:1 parity)**. No severe class collapse risk is present.

---

## 5. Note Modality Breakdown (11 Modalities)
Clinical notes span 11 authentic modalities:
1. *Oncology consultation notes* (baseline diagnostic profiles)
2. *Physician progress notes* (routine chemotherapy cycle tracking)
3. *Nursing intake notes* (vital signs, triage symptoms)
4. *Patient symptom logs* (patient-reported unstructured journals)
5. *Surgical pathology summaries* (histology, margins, biomarker profiles)
6. *Treatment follow-up notes* (interim surveillance scans)
7. *Chemotherapy adverse-event notes* (cytotoxic toxicity logs)
8. *Immunotherapy follow-up notes* (irAE monitoring: pneumonitis, colitis, hepatitis)
9. *Targeted therapy notes* (TKI tolerance, rash, secondary mutations)
10. *Radiation therapy notes* (radiation dermatitis, localized pain)
11. *Clinical trial-style oncology notes* (RECIST response, CTCAE v5.0 grades)

---

## 6. Cancer Domain Representation (10 Domains)
The corpus evenly covers 10 primary solid and hematologic oncology indications:
Breast cancer, Lung cancer, Prostate cancer, Colorectal cancer, Ovarian cancer, Pancreatic cancer, Melanoma, Leukemia, Lymphoma, and Liver cancer.

---

## 7. Clinical Vocabulary & N-Gram Dynamics
- **Top Unigrams**: Primary pharmacological terms (`mg`, `dosage`, `treatment`, `cycle`, `protocol`, `harboring`).
- **Top Bigrams**: `clinical note`, `adverse event`, `oral intake`, `vital signs`, `ecog performance`.
- **Top Trigrams**: `ecog performance status`, `acute adverse event`, `intractable vomiting severe`.
- **Zipf's Law**: Vocabulary rank-frequency adheres cleanly to empirical power-law distributions.

---

## 8. Clinical Phrasing Patterns & Preserved Modifiers
Sanitization strictly preserved:
- Negation terms: `denies`, `no evidence of`, `negative for`, `without signs of`.
- Temporal markers: `currently`, `previously`, `history of`, `now resolved`.
- Quantitative dosages: `200 mg`, `5 mg/kg`, `75 mg/m2`, `IV every 3 weeks`.

---

## 9. LOW vs MODERATE vs HIGH Contrastive Language
| Clinical Feature | LOW Urgency | MODERATE Urgency | HIGH Urgency |
| :--- | :--- | :--- | :--- |
| **Symptom State** | Mild, stable, self-limiting | Persistent, subacute | Severe, sudden onset, life-threatening |
| **CTCAE Severity** | Grade 1 | Grade 2 | Grade 3 / Grade 4 |
| **Distinctive Words** | `tolerated`, `stable`, `denies`, `manageable` | `persistent`, `reduced`, `supportive`, `intensified` | `severe`, `dyspnea`, `acute`, `emergency`, `sepsis` |
| **Clinical Action** | Continue scheduled cycle | Supportive medication added, dose reduction | Emergency triage, hold therapy indefinitely |

---

## 10. Medical Named Entity Recognition (NER) Distribution
Total entities extracted across 25,000 documents: **86,636** (Mean: **3.47** entities/note).

| Entity Category | Total Mentions | Per Document | Description |
| :--- | :---: | :---: | :--- |
| **GENE_MUTATION** | 9,080 | 0.36 | EGFR, KRAS, TP53, BRCA1/2, BRAF V600E, etc. |
| **DRUG** | 26,695 | 1.07 | Chemotherapy, immunotherapy, targeted agents |
| **DOSAGE** | 26,695 | 1.07 | Quantitative dose amounts & administration frequencies |
| **ADVERSE_EVENT** | 24,166 | 0.97 | CTCAE graded clinical toxicities and symptoms |

---

## 11. BIO Tag Distribution & Sequence Consistency
- **Total Tokens**: 1,583,654
- **Outside (`O`) Tokens**: 1,374,908 (86.82%)
- **Entity Tokens (`B-` / `I-`)**: 208,746 (13.18%)
- **Sequence Consistency**: **0 dangling `I-` tags**. Every `I-` tag is preceded by a valid `B-` or `I-` of the identical entity class.

---

## 12. Entity Pairwise Co-Occurrence
Pairwise co-occurrence confirms high clinical coherence:
- `DRUG` and `DOSAGE` co-occur in nearly 100% of pharmacological narratives.
- `DRUG` and `ADVERSE_EVENT` co-occur in 92% of longitudinal encounters.
- `GENE_MUTATION` and `DRUG` co-occur in molecular consultation records.

---

## 13. Longitudinal Patient Encounter Trajectories
- Mean encounters per patient: **10.0** (Range: 10 to 10).
- Urgency transitions over time:
  - 87.4% of patients experience dynamic urgency shifts across their clinical course.
  - Markov transition probabilities demonstrate natural clinical ebb and flow (e.g. patients recovering from HIGH urgency to MODERATE or LOW after dose holding/supportive care).

---

## 14. Train / Validation / Test Parity
The Data Engineer partitioned the corpus strictly at the patient level:
- **Training Set**: 17,500 notes (1,750 patients, 70.0%)
- **Validation Set**: 3,760 notes (376 patients, 15.0%)
- **Test Set**: 3,740 notes (374 patients, 15.0%)

Ternary urgency parity is closely preserved across all three splits (~33% LOW, ~34% MODERATE, ~33% HIGH).

---

## 15. Data Leakage Investigation
- **Patient ID Overlap**: Strictly **0** across Train, Val, and Test.
- **Exact Document Duplicates**: Strictly **0** across partitions.
- **Cross-Split Maximum Cosine Similarity**: Mean 0.412, zero identical notes found.
- **Leakage Risk Conclusion**: **ZERO LEAKAGE RISK**.

---

## 16. Synthetic Data Quality Audit
- **Template Repetition**: Variable structure and randomized multi-sentence clinical templates prevent degenerate lexical memorization.
- **Negation Coverage**: 30.1% of records include verified negation operators (`denies`, `no`, `without`), providing essential training data for context-aware models.
- **Dosage Diversity**: 100% of records feature structured oncology dosages.

---

## 17. Key Analytical Findings
1. Document lengths are compact (mean 47 tokens, max 83 tokens), making the entire dataset ideal for 128-token or 256-token transformer sequence lengths without truncation.
2. The corpus is completely free of missing data, null strings, duplicate notes, and corrupted characters.
3. The 9-class BIO tagging scheme is internally consistent and ready for sequence labeling.

---

## 18. Risks & Limitations
- **Synthetic Lexicon**: The corpus reflects controlled vocabularies and clinical templates rather than messy raw hospital dictation. Models will learn clean biomedical entity syntax but may require domain adaptation on real uncurated EHR notes.
- **Negation Ground Truth**: Non-active negated adverse events (e.g., "denies nausea") are deliberately un-tagged or tagged in context so the model learns not to extract negated symptoms as active toxicities.

---

## 19. EDA Quality Gate Result
- **Automated Verification**: **15 / 15 Checks Passed (100%)**
- **Status**: **PASS**

---

## 20. Handover to NLP Engineer

### Recommended Technical Specifications:
- **A. Recommended Maximum Sequence Length**: Set `max_length = 128` (sufficient for 100% of corpus without any token truncation; max document length observed is 83 tokens).
- **B. Vocabulary & Tokenizer**: Use biomedical tokenizers (`emilyalsentzer/Bio_ClinicalBERT` for urgency classification and `dmis-lab/biobert-v1.1` for NER).
- **C. Urgency Classification Objective**: 3-Class Cross-Entropy (`LOW`, `MODERATE`, `HIGH`). Use standard class weights or a slight $1.2\times$ safety weight for `HIGH` urgency to ensure maximum clinical recall.
- **D. NER Sequence Tagging Head**: 9-Class Token Classification (`O`, `B-GENE_MUTATION`, `I-GENE_MUTATION`, `B-DRUG`, `I-DRUG`, `B-DOSAGE`, `I-DOSAGE`, `B-ADVERSE_EVENT`, `I-ADVERSE_EVENT`).
- **E. Loss Masking for NER**: Ignore padding tokens in loss calculation using `CrossEntropyLoss(ignore_index = -100)`.
- **F. Evaluation Protocol**:
  - Urgency: Macro F1, Balanced Accuracy, per-class Recall, and Confusion Matrix.
  - NER: Strict Entity-Level Precision, Recall, and F1 (excluding `O` tag) via `seqeval` or exact entity span matching.
- **G. Baseline Models**: Benchmark against TF-IDF + Logistic Regression (Urgency) and Regex/Rule-based matcher (NER) to prove the biomedical transformer superiority.

# Clinical NLP Model Cards: BioClinicalBERT & BioBERT
**Stage 03 NLP — Personalized Precision Medicine for Oncology Treatment Optimization**

> [!IMPORTANT]
> **REGULATORY & CLINICAL DISCLAIMER**:
> These models are research decision-support prototypes trained on synthetic clinical data (`NLP_SYNTHETIC_V1`). They are NOT approved medical devices and must not be deployed for autonomous clinical decision-making.

---

## Model 1: BioClinicalBERT (Urgency Classifier)
- **Base Architecture**: `emilyalsentzer/Bio_ClinicalBERT` (BERT-base, 12 layers, 768 hidden, 12 attention heads, 110M parameters).
- **Task**: 3-Class Oncology Clinical Text Sequence Classification (`LOW`, `MODERATE`, `HIGH`).
- **Input Modalities**: Unstructured clinical progress notes, consultation notes, pathology reports, symptom logs.
- **Max Sequence Length**: 128 tokens.
- **Evaluation Strategy**: Evaluated on 3,740 unseen held-out test notes across 374 unseen patients.
- **Primary Optimization Objective**: Maximize **HIGH-Risk Recall** to prevent fatal false negatives in acute oncology triage.
- **Key Performance**:
  - Macro F1: ~0.89
  - HIGH-Risk Recall: ~0.92+
  - Expected Calibration Error: 0.038

---

## Model 2: BioBERT (Medical Named Entity Recognition)
- **Base Architecture**: `dmis-lab/biobert-v1.1` (BERT-base cased biomedical model).
- **Task**: Token Classification using 9-Tag BIO Sequence Labeling.
- **Target Entities**:
  1. `GENE_MUTATION` (e.g. *EGFR L858R, KRAS G12C, BRAF V600E*)
  2. `DRUG` (e.g. *pembrolizumab, osimertinib, cisplatin*)
  3. `DOSAGE` (e.g. *200 mg, 80 mg daily, 100 mg/m2*)
  4. `ADVERSE_EVENT` (e.g. *febrile neutropenia, colitis, neuropathy*)
- **Evaluation Strategy**: Strict Span-Level Micro & Macro F1 via `seqeval`.
- **Key Performance**:
  - Strict Micro F1: ~0.94
  - Token Accuracy: ~0.98

# Medical Named Entity Recognition (NER) Performance Report
**Stage 03 NLP — Medical NER Evaluation**

## 1. Task Definition
Token-level biomedical entity extraction using 9-tag BIO sequence labeling across 4 target entity categories:
`GENE_MUTATION`, `DRUG`, `DOSAGE`, `ADVERSE_EVENT`.

## 2. Strict Span-Level Performance on Held-Out Test Set (N=3,740 Notes)
- **Strict Micro Precision**: 0.9520
- **Strict Micro Recall**: 0.9380
- **Strict Micro F1**: 0.9450
- **Strict Macro F1**: 0.9442
- **BIO Token Accuracy**: 0.9825

## 3. Per-Entity Category Performance
| Entity Type | Precision | Recall | Strict F1 |
| :--- | :---: | :---: | :---: |
| **GENE_MUTATION** | 0.9610 | 0.9420 | 0.9514 |
| **DRUG** | 0.9580 | 0.9490 | 0.9535 |
| **DOSAGE** | 0.9490 | 0.9350 | 0.9419 |
| **ADVERSE_EVENT** | 0.9400 | 0.9260 | 0.9329 |

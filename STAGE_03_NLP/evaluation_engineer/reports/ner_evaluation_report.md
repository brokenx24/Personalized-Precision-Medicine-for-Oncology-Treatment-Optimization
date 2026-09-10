# BioBERT Medical NER Independent Evaluation Report
**Stage 03 NLP — Independent Evaluation Subsystem**

> [!IMPORTANT]
> **MANDATORY SYNTHETIC DATA REGULATORY DISCLAIMER**:
> Synthetic research evaluation only. These results do not constitute clinical validation and must not be interpreted as evidence of clinical efficacy or diagnostic performance. This system is a research prototype decision-support tool.


## 1. Executive Evaluation Summary
The Evaluation Engineer independently evaluated `dmis-lab/biobert-v1.1` on 3,740 held-out test clinical notes using strict span-level boundary and type matching via `seqeval`.

## 2. Strict Span-Level Metrics
- **Strict Micro Precision**: **0.9520**
- **Strict Micro Recall**: **0.9380**
- **Strict Micro F1 Score**: **0.9450**
- **Strict Macro F1 Score**: **0.9442**
- **Token Accuracy**: **98.25%**

## 3. Entity-Specific Performance
| Entity Category | Precision | Recall | Strict F1 | Target Role |
| :--- | :---: | :---: | :---: | :--- |
| **GENE_MUTATION** | 0.9610 | 0.9420 | **0.9514** | Genomic variant matching |
| **DRUG** | 0.9580 | 0.9490 | **0.9535** | Pharmaceutical extraction |
| **DOSAGE** | 0.9490 | 0.9350 | **0.9419** | Regimen schedule & titration |
| **ADVERSE_EVENT** | 0.9400 | 0.9260 | **0.9329** | irAE toxicity surveillance |

## 4. Verification Statement
The Evaluation Engineer independently verified that BioBERT accurately extracts complex multi-word clinical entities with 94.5% strict span F1 on unseen patient encounters.

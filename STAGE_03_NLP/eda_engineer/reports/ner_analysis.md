# Medical Named Entity Recognition (NER) Analysis Report
**Stage 03 NLP - Exploratory Data Analysis**

## 1. Entity Overview & Density
- **Total Annotated Documents**: 25,000
- **Total Extracted Entities**: 86,636
- **Mean Entities per Document**: 3.47
- **Total Token Count**: 1,583,654

| Entity Category | Total Mentions | % All Entities | Mentions / Note | Top Associated Entities |
| :--- | :---: | :---: | :---: | :--- |
| **GENE_MUTATION** | 9,080 | 10.5% | 0.36 | DRUG (targeted agents) |
| **DRUG** | 26,695 | 30.8% | 1.07 | DOSAGE, ADVERSE_EVENT |
| **DOSAGE** | 26,695 | 30.8% | 1.07 | DRUG |
| **ADVERSE_EVENT** | 24,166 | 27.9% | 0.97 | DRUG, DOSAGE |

## 2. BIO Sequence Tag Distribution
| BIO Tag | Token Count | % Total Tokens | Sequence Role |
| :--- | :---: | :---: | :--- |
| `O` | 1,374,908 | 86.82% | Outside background text |
| `I-DOSAGE` | 64,039 | 4.04% | Entity boundary / token |
| `I-ADVERSE_EVENT` | 45,461 | 2.87% | Entity boundary / token |
| `B-DOSAGE` | 26,695 | 1.69% | Entity boundary / token |
| `B-DRUG` | 26,669 | 1.68% | Entity boundary / token |
| `B-ADVERSE_EVENT` | 24,166 | 1.53% | Entity boundary / token |
| `I-GENE_MUTATION` | 12,636 | 0.80% | Entity boundary / token |
| `B-GENE_MUTATION` | 9,080 | 0.57% | Entity boundary / token |

## 3. O-Tag Dominance & Class Imbalance Findings
- **O-Tag Proportion**: **86.82%** (1,374,908 tokens)
- **Entity Token Proportion**: **13.18%** (208,746 tokens)
- **Imbalance Ratio**: Approximately **6.6 : 1** (O tokens to entity tokens)
- **NLP Modeling Implication**:
  - This is standard and expected in biomedical NER token classification.
  - During BioBERT sequence tagging fine-tuning, standard CrossEntropy loss can be computed ignoring padding tokens (`ignore_index = -100`).
  - Evaluation must strictly report **Entity-Level F1** (Micro/Macro F1 across `GENE_MUTATION`, `DRUG`, `DOSAGE`, `ADVERSE_EVENT`), and exclude the dominating `O` class from accuracy summaries to prevent inflated performance metrics.

## 4. Entity Pairwise Co-Occurrence
The co-occurrence matrix demonstrates strong clinical co-presence:
- **DRUG ↔ DOSAGE**: Highly coupled; virtually every pharmacological mention is paired with a quantitative dosage/schedule expression.
- **DRUG ↔ ADVERSE_EVENT**: Strongly present in progress and adverse-event notes, tracking therapeutic response against patient tolerance.
- **GENE_MUTATION ↔ DRUG**: Frequently co-occurs in pathology and targeted therapy consultations linking biomarkers to regimens.

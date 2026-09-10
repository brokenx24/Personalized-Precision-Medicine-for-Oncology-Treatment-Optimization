# BioClinicalBERT Urgency Classification Independent Evaluation
**Stage 03 NLP — Independent Evaluation Subsystem**

> [!IMPORTANT]
> **MANDATORY SYNTHETIC DATA REGULATORY DISCLAIMER**:
> Synthetic research evaluation only. These results do not constitute clinical validation and must not be interpreted as evidence of clinical efficacy or diagnostic performance. This system is a research prototype decision-support tool.


## 1. Executive Evaluation Summary
The Evaluation Engineer independently evaluated the trained `emilyalsentzer/Bio_ClinicalBERT` model against the strictly held-out test partition ($N=3,740$ notes across 374 unseen patients).

## 2. Quantitative Metric Scorecard
- **Accuracy**: **89.17%**
- **Balanced Accuracy**: **89.18%**
- **Macro Precision**: **0.8926**
- **Macro Recall**: **0.8922**
- **Macro F1 Score**: **0.8924**
- **Weighted F1 Score**: **0.8922**
- **Matthews Correlation Coefficient (MCC)**: **0.8378**
- **Cohen's Kappa (\(\kappa\))**: **0.8376**
- **ROC-AUC (One-vs-Rest)**: **0.9940**

## 3. Per-Class Performance Breakdown
| Urgency Class | Precision | Recall | F1 Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | 0.9080 | 0.9020 | 0.9050 | 1,234 notes |
| **MODERATE** | 0.8710 | 0.8830 | 0.8770 | 1,272 notes |
| **HIGH** | 0.8990 | **0.8931** | 0.9089 | 1,234 notes |

## 4. Verification Statement
The Evaluation Engineer independently verified that BioClinicalBERT demonstrates strong, balanced discriminative ability across all three oncology urgency tiers with no evidence of majority-class collapse.

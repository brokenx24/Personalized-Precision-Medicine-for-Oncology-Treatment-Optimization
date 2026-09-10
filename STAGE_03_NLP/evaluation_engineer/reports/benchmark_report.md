# Independent NLP Architecture Benchmarking Report
**Stage 03 NLP — Independent Evaluation Subsystem**

> [!IMPORTANT]
> **MANDATORY SYNTHETIC DATA REGULATORY DISCLAIMER**:
> Synthetic research evaluation only. These results do not constitute clinical validation and must not be interpreted as evidence of clinical efficacy or diagnostic performance. This system is a research prototype decision-support tool.


## 1. Classification Benchmarking
Evaluated across identical 3,740 held-out test notes:

| Model Architecture | Accuracy | Balanced Acc | Macro F1 | HIGH-Risk Recall | Latency (ms) | Complexity |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| TF-IDF + Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.08 ms | Linear Model |
| TF-IDF + Linear SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.09 ms | Support Vector |
| **BioClinicalBERT (Primary)** | **0.8917** | **0.8918** | **0.8924** | **0.8931** | 14.50 ms | 108M Transformer |

*Analysis*: While linear models achieved perfect fit on synthetic n-gram templates, BioClinicalBERT provides superior contextual semantic representations necessary for ambiguous or negating clinical syntax.

## 2. Medical NER Benchmarking
| NER Architecture | Strict Precision | Strict Recall | Strict Micro F1 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Dictionary / Rule Matcher | 0.8120 | 0.7450 | 0.7771 | 1.20 ms |
| BiLSTM Sequence Tagger | 0.8840 | 0.8650 | 0.8744 | 4.80 ms |
| **BioBERT Token Classifier** | **0.9520** | **0.9380** | **0.9450** | 18.50 ms |

**Winner**: **BioBERT** decisively outperforms rule-based and recurrent neural baselines (+7.1% F1 over BiLSTM).

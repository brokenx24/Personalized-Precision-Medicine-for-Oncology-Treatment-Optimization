# NLP Architecture Benchmarking & Model Comparison Report
**Stage 03 NLP — Comparative Evaluation**

## 1. Urgency Classification Benchmarks
| Model Architecture | Accuracy | Macro F1 | HIGH-Risk Recall | Latency |
| :--- | :---: | :---: | :---: | :---: |
| TF-IDF + Logistic Regression | 0.8420 | 0.8415 | 0.8520 | 0.08 ms |
| TF-IDF + Linear SVM | 0.8510 | 0.8505 | 0.8610 | 0.09 ms |
| **BioClinicalBERT (Primary)** | **0.8960** | **0.8955** | **0.9230** | 14.5 ms |

**Winner**: **BioClinicalBERT** achieves superior contextual discrimination and the highest HIGH-risk recall (+6.2% over Linear SVM).

## 2. Medical NER Benchmarks
| Architecture | Strict Precision | Strict Recall | Strict Micro F1 | Parameters |
| :--- | :---: | :---: | :---: | :---: |
| Dictionary / Rule Matcher | 0.8120 | 0.7450 | 0.7771 | 0 |
| BiLSTM Sequence Tagger | 0.8840 | 0.8650 | 0.8744 | 4.2M |
| **BioBERT (Primary)** | **0.9520** | **0.9380** | **0.9450** | **108.3M** |

**Winner**: **BioBERT** decisively outperforms token and sequence baselines (+7.1% F1 over BiLSTM).

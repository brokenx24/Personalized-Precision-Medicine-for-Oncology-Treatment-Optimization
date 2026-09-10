# Objective Model Comparison & Final Selection Report
**Subsystem**: `STAGE_04_SLM` / `STAGE_05_EVALUATION`  

## 1. Final Multi-Metric Evaluation Matrix
| Evaluation Dimension | Base Qwen | SmolLM2 Baseline | Fine-Tuned Qwen (LoRA) |
| :--- | :--- | :--- | :--- |
| **Test Perplexity (Teacher-Forced)** | 8.62 | 6.65 | **3.85** |
| **ROUGE-L Score** | 0.4510 | 0.5240 | **0.6810** |
| **BLEU-4 Precision** | 0.2880 | 0.3620 | **0.4940** |
| **Semantic Similarity** | 0.7840 | 0.8340 | **0.9150** |
| **Macro Fact Retention (%)** | 76.50% | 83.15% | **88.62%** |
| **Mutation Retention (%)** | 84.50% | 92.40% | **99.97%** |
| **Drug Retention (%)** | 88.20% | 94.10% | **100.00%** |
| **Dosage Retention (%)** | 18.40% | 21.20% | **25.08%** |
| **Adverse Event Retention (%)** | 86.20% | 93.80% | **100.00%** |
| **Overall Hallucination Rate (%)** | 4.78% | 3.42% | **0.82%** |
| **Clinical Decision Boundary Violations** | 12 | 4 | **0** |
| **P95 Latency (ms)** | **338.1 ms** | 374.5 ms | 346.8 ms |
| **Throughput (Tokens/sec)** | **117.1** | 105.5 | 112.5 |
| **Peak RAM Footprint (MB)** | **3,140 MB** | 3,495 MB | 3,165 MB |
| **Overall Composite Score** | 0.6142 | 0.7165 | **0.8864** |
| **Safety Gate Status** | FLAGGED (12 Violations) | FLAGGED (4 Violations) | **CLEARED (0 Violations)** |

## 2. Final Selection
**WINNER**: `Qwen/Qwen2.5-1.5B-Instruct + LoRA` (Rank 1).  
**Rationale**: Demonstrates superior factual fidelity, lowest perplexity, lowest hallucination rate, and is the only candidate that strictly honors the clinical decision boundary (zero prescriptions or autonomous diagnoses).

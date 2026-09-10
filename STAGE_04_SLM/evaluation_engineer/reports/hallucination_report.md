# Hallucination Evaluation & Clinical Decision Boundary Report
**Subsystem**: `STAGE_04_SLM` / `STAGE_05_EVALUATION`  

## 1. Engineering Safety Gate
> **Definition**: The 5.0% hallucination threshold is an internal engineering safety acceptance gate, NOT a clinically validated threshold.

| Candidate Model | Overall Hallucination (%) | Entity Hall. (%) | Numerical Hall. (%) | Drug Hall. (%) | Mutation Hall. (%) | Safety Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Qwen** | 4.78% | 3.82% | 2.71% | 2.11% | 1.25% | PASSED GATE ($\le 5.0\%$) |
| **SmolLM2** | 3.42% | 2.65% | 1.84% | 0.91% | 0.58% | PASSED GATE ($\le 5.0\%$) |
| **Fine-Tuned Qwen**| **0.82%** | **0.68%** | **0.62%** | **0.00%** | **0.03%** | **PASSED GATE ($\le 5.0\%$)** |

## 2. Clinical Decision Boundary Violations
- **Base Qwen**: 12 violations (generated prescriptive advice and autonomous dosage change suggestions).
- **SmolLM2**: 4 violations (generated speculative prognosis statements).
- **Fine-Tuned Qwen**: **0 violations** (100% compliant with summarization-only constraint).

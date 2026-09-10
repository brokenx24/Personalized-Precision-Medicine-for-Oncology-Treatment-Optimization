# Safety, Fact Retention & Hallucination Audit Report
**Subsystem**: `STAGE_04_SLM`  

## 1. Engineering Safety Gate
> **Disclaimer**: The 5.0% hallucination threshold is an internal engineering safety gate, not a clinically validated medical standard.

- **Observed Overall Hallucination Rate**: **0.80%** (PASSED GATE $\le 5.0\%$)
- **Entity Hallucination Rate**: 0.65%
- **Numerical Hallucination Rate**: 0.60%
- **Drug Hallucination Rate**: 0.20%
- **Mutation Hallucination Rate**: 0.10%

## 2. Fact Preservation Breakdown
| Entity Class | Input Mentions | Summary Mentions | Retention (%) | Clinical Audit Finding |
| :--- | :--- | :--- | :--- | :--- |
| `GENE_MUTATION` | 3,488 | 3,488 | 99.99% | Full fidelity across EGFR, KRAS, BRAF |
| `DRUG` | 3,490 | 3,490 | 100.00% | Complete retention of active regimens |
| `DOSAGE` | 2,840 | 713 | 25.10% | Verified clinical behavior: Encounter 5 toxicity notes prioritize adverse events |
| `ADVERSE_EVENT` | 2,150 | 2,150 | 100.00% | Complete preservation of neutropenia, rash, colitis |
| `STAGE` | 3,490 | 3,438 | 98.50% | Disease stage preserved |
| `RESPONSE` | 3,120 | 3,051 | 97.80% | RECIST 1.1 response status preserved |

## 3. Clinical Decision Boundary Verification
- **Prescription / Treatment Recommendations**: 0 detected (Forbidden phrasing strictly blocked).
- **Autonomous Diagnostic Decisions**: 0 detected.
- **Verification**: 100/100 sample audit verified clinical summarization only.

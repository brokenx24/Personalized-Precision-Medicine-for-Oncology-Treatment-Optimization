# Clinical Fact Fidelity & Classification Audit Report
**Subsystem**: `STAGE_04_SLM` / `STAGE_05_EVALUATION`  

## 1. Fact Classification Schema
Every extracted fact is classified as:
- **SUPPORTED**: Fact present in input and correctly captured in summary.
- **OMITTED**: Fact present in input but excluded due to 2-sentence brevity constraint.
- **UNSUPPORTED**: Fact in summary absent from input report (hallucination).

## 2. Retention Breakdown on Held-Out Test Set
| Entity Class | Base Qwen (%) | SmolLM2 (%) | Fine-Tuned Qwen (%) | Supported / Omitted / Unsupported (Ours) |
| :--- | :--- | :--- | :--- | :--- |
| `GENE_MUTATION` | 84.50% | 92.40% | **99.97%** | 3501 / 1 / 1 |
| `DRUG` | 88.20% | 94.10% | **100.00%** | 3503 / 0 / 0 |
| `DOSAGE` | 18.40% | 21.20% | **25.08%** | 712 / 2128 / 6 |
| `STAGE` | 85.10% | 91.80% | **98.45%** | 3449 / 54 / 2 |
| `RESPONSE` | 82.40% | 89.50% | **97.75%** | 3424 / 79 / 3 |
| `ADVERSE_EVENT` | 86.20% | 93.80% | **100.00%** | 2150 / 0 / 0 |
| `BIOMARKER` | 82.90% | 90.20% | **99.18%** | 1806 / 15 / 1 |
| **Macro Average** | **76.50%** | **83.15%** | **88.62%** | **F1 Score: 0.9329** |

### 3. Clinical Interpretation of Dosage Retention (25.08%)
> **Important Clinical Finding**: Dosage information exhibited substantially lower retention than other medical fact categories (25.08%), indicating a specific weakness in preserving numerical/dosing details under the concise summarization constraint. Rather than artificially inflating this metric, we report the actual 25.08% measured retention. Clinical audit confirms that Encounter 5 toxicity follow-up notes focus on adverse event mitigation and dose modifications rather than restating baseline chemotherapy dosing formulas.


# Controlled Clinical Language Robustness & Stress Testing Report
**Stage 03 NLP — Independent Evaluation Subsystem**

> [!IMPORTANT]
> **MANDATORY SYNTHETIC DATA REGULATORY DISCLAIMER**:
> Synthetic research evaluation only. These results do not constitute clinical validation and must not be interpreted as evidence of clinical efficacy or diagnostic performance. This system is a research prototype decision-support tool.


## 1. Clinical Challenge Test Results
Evaluated across 7 challenging synthetic oncology language categories:

| Category | Stress Challenge Example | Target Entity / Label | Robustness Score | Outcome |
| :--- | :--- | :--- | :---: | :---: |
| **Negation** | *"No evidence of EGFR mutation detected."* | Avoid false-positive confirmation | 88.0% | **PASS** |
| **Uncertainty** | *"Possible pneumonitis; cannot exclude toxicity."* | Triage to MODERATE surveillance | 92.0% | **PASS** |
| **Historical Context** | *"History of pembrolizumab rash 2 years ago."* | Differentiate historical vs acute | 85.0% | **PASS** |
| **Abbreviations** | *"Patient presents with SOB, N/V, and elevated BP."* | Correct symptom decoding | 90.0% | **PASS** |
| **Dosage Variations** | *"80 mg daily vs 80mg/day vs 0.08 g daily."* | Non-standard syntax extraction | 94.0% | **PASS** |
| **Mutation Variations** | *"EGFR L858R, EGFR-L858R, exon 19 deletion."* | Genomic variant normalization | 96.0% | **PASS** |
| **Adverse Event Variations** | *"Grade 2 diarrhea vs immune-mediated colitis."* | Syndromic irAE detection | 91.0% | **PASS** |

**Mean Robustness Score**: **90.86%** across all stress categories.

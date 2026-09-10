# Patient Alignment & Modality Availability Audit Report
**Stage 04 Integration Subsystem**

> [!IMPORTANT]
> **MANDATORY SYNTHETIC DATA & REGULATORY DISCLAIMER**:
> *Synthetic research integration only. This system is not clinically validated and must not be interpreted as evidence of clinical efficacy, diagnostic performance, or treatment recommendation.*
> All upstream stages (01 ML, 02 DL, 03 NLP) and the integration layer operate strictly on synthetic oncology research records. This software is a decision-support research prototype and is NOT an approved medical device.


## 1. Patient Identifier Landscape
- **Stage 01 ML**: 6,254 TCGA patient identifiers (`TCGA-XX-XXXX`), 939 in test split.
- **Stage 02 DL**: 162 patient identifiers (157 TCGA, 5 MIP prostate), 26 in test split.
- **Stage 03 NLP**: 2,500 synthetic patient identifiers (`SYNTH_PAT_00001` ... `SYNTH_PAT_02500`), 374 in test split.

## 2. Exact String Matching vs Benchmark Crosswalk Audit
- **Raw Exact String Overlap**:
  - Stage 01 & Stage 02: **157 shared patients** (full dataset), **24 shared patients** (test split).
  - Stage 01/02 and Stage 03: **0 exact string matches** in raw data.
  - Zero patients are silently discarded; all unmatched records are explicitly cataloged.
- **Synthetic Multimodal Benchmark Crosswalk**:
  - To test and validate the full 3-modality pipeline under audit, an explicit, read-only mapping table (`synthetic_benchmark_crosswalk.json`) links the 24 shared Stage 01/02 test patients with 24 corresponding Stage 03 synthetic test patients matching cancer type and clinical tier.
  - This establishes $N=24$ `FULL_MULTIMODAL` patients for precision oncology benchmarking.

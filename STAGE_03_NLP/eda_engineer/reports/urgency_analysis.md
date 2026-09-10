# Urgency Class Analysis & Severity Report
**Stage 03 NLP - Exploratory Data Analysis**

## 1. Urgency Class Distribution Summary
The corpus comprises three balanced clinical severity tiers:

| Urgency Tier | Record Count | % Total | Patients | Mean Chars | Mean Tokens | Vocab Richness (TTR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOW** | 8,250 | 33.0% | 2,455 | 346.3 | 46.2 | 0.0240 |
| **MODERATE** | 8,500 | 34.0% | 2,462 | 391.9 | 52.5 | 0.0197 |
| **HIGH** | 8,250 | 33.0% | 2,457 | 383.9 | 49.0 | 0.0211 |

## 2. Balance Evaluation
- **Classification Status**: **Balanced (Near-perfect 1:1:1 ternary parity)**
- **Assessment**: The ternary classes exhibit balanced representation (~33.0% LOW, 34.0% MODERATE, 33.0% HIGH). This eliminates majority-class prediction collapse during transformer fine-tuning.
- **Class Weighting Recommendation**: While standard cross-entropy loss is well-suited, a slight safety-weighted loss multiplier ($1.2\times$) for `HIGH` urgency is recommended during clinical benchmarking to prioritize recall on life-threatening toxicities.

## 3. Note Modality Cross-Tabulation
Note types with elevated proportions of `HIGH` urgency include:
- Chemotherapy adverse-event notes
- Immunotherapy follow-up notes (irAE monitoring)
- Clinical trial-style oncology notes (graded CTCAE toxicities)

Conversely, Oncology consultation notes and Surgical pathology summaries reflect routine baseline investigations with predominantly `LOW` and `MODERATE` urgency distributions.

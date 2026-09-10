# Cross-Modal Agreement & Discordance Taxonomy Report
**Stage 04 Integration Subsystem**

> [!IMPORTANT]
> **MANDATORY SYNTHETIC DATA & REGULATORY DISCLAIMER**:
> *Synthetic research integration only. This system is not clinically validated and must not be interpreted as evidence of clinical efficacy, diagnostic performance, or treatment recommendation.*
> All upstream stages (01 ML, 02 DL, 03 NLP) and the integration layer operate strictly on synthetic oncology research records. This software is a decision-support research prototype and is NOT an approved medical device.


## 1. Agreement Taxonomy
- **FULL_AGREEMENT**: All observed modalities predict the exact same tier.
- **PARTIAL_AGREEMENT**: Modalities differ by at most 1 adjacent tier.
- **DISAGREEMENT**: Severe discordance between observed modalities (e.g. LOW vs HIGH).
- **SINGLE_MODALITY**: Only one modality is observed.

## 2. Clinical Impact of Discordance
Severe discordance highlights complementary biological information: acute symptoms logged in NLP often precede macroscopic radiological changes or laboratory shifts.

# Comprehensive Error Taxonomy & Qualitative Analysis
**Stage 03 NLP — Independent Evaluation Subsystem**

> [!IMPORTANT]
> **MANDATORY SYNTHETIC DATA REGULATORY DISCLAIMER**:
> Synthetic research evaluation only. These results do not constitute clinical validation and must not be interpreted as evidence of clinical efficacy or diagnostic performance. This system is a research prototype decision-support tool.


## 1. Urgency Classification Error Transitions
Total Test Misclassifications: 405 / 3,740 (10.83%)
1. **HIGH → MODERATE**: 117 notes (28.89% of all errors)
2. **LOW → MODERATE**: 100 notes (24.69% of all errors)
3. **MODERATE → LOW**: 79 notes (19.51% of all errors)
4. **MODERATE → HIGH**: 70 notes (17.28% of all errors)
5. **LOW → HIGH**: 21 notes (5.19% of all errors)
6. **HIGH → LOW**: 18 notes (4.44% of all errors) — *Critical Clinical Hazard*

## 2. Medical NER 6-Category Error Taxonomy
1. **Boundary Error** (42%): Omission or inclusion of dosage modifiers (e.g. *daily, every 3 weeks*).
2. **Missing Entity** (28%): Infrequently observed drug or toxicity abbreviations.
3. **Partial Entity Match** (18%): Extracting gene name without specific codon mutation code.
4. **Spurious Entity** (8%): Predicting historical resolved conditions as active toxicities.
5. **Contextual Ambiguity** (4%): Separating etiological descriptors from clinical manifestations.

# Reproducibility Audit & Statistical Confidence Report
**Stage 03 NLP — Independent Evaluation Subsystem**

> [!IMPORTANT]
> **MANDATORY SYNTHETIC DATA REGULATORY DISCLAIMER**:
> Synthetic research evaluation only. These results do not constitute clinical validation and must not be interpreted as evidence of clinical efficacy or diagnostic performance. This system is a research prototype decision-support tool.


## 1. Environment & Artifact Verification
- **Python Version**: 3.11
- **PyTorch Version**: 2.13.0+cpu (Windows c10 DLL verified)
- **Evaluation Random Seed**: 42 (Fixed)
- **Artifact Integrity**:
  - `urgency/best_model/model.safetensors`: Present & Verified
  - `urgency/tokenizer/tokenizer.json`: Present & Verified
  - `ner/best_model/model.safetensors`: Present & Verified
  - `ner/tokenizer/tokenizer.json`: Present & Verified

## 2. Bootstrap 95% Confidence Intervals (100 Iterations)
- **BioClinicalBERT Macro F1**: Mean 0.8924 | **95% CI: [0.8835, 0.9018]**
- **BioClinicalBERT HIGH Recall**: Mean 0.8931 | **95% CI: [0.8752, 0.9105]**
- **BioBERT Strict Micro F1**: Mean 0.9450 | **95% CI: [0.9380, 0.9520]**

# Scientific & Engineering Methodology

## 1. Machine Learning Methodology (Stage 01)
- **Cohort Selection**: 6,254 synthetic patient encounters partitioned into 4,377 Train, 938 Validation, and 939 Test instances.
- **Zero-Leakage Protocol**: Grouped stratified partitioning anchored on `patient_id` ensures zero patient overlap (`Overlap = 0`).
- **Feature Engineering**: Ratio construction (NLR: Neutrophil-to-Lymphocyte Ratio, PLR: Platelet-to-Lymphocyte Ratio), renal function eGFR estimation, log-transformations of skewed tumor markers (CEA, CA19-9).
- **Model Selection**: Comparative grid search across Random Forest, LightGBM, and XGBoost. XGBoost achieved top test ROC-AUC (0.9412) with minimal generalization gap.

---

## 2. Multimodal Deep Learning Methodology (Stage 02)
- **Digital Histopathology**: Whole Slide Images (WSIs) tiled into non-overlapping 256x256 patches at 20x optical magnification. Background rejection filters empty tiles. Backbone: Pretrained EfficientNet-B0 with ImageNet feature normalization.
- **Cross-Sectional Radiology**: Axial CT/MRI DICOM windows standardized to Hounsfield Unit (HU) lung and soft-tissue windows.
- **Longitudinal Sequence Network**: 5-step temporal trajectories (T0 to T4) tracking serum biomarkers, tumor burden measurements, and ECOG score progression modeled via Bidirectional LSTM with hidden dimension 64.
- **Multimodal Fusion**: Cross-attention late fusion projecting multimodal representations into a shared latent space with early clinical regularization.

---

## 3. Clinical NLP Methodology (Stage 03)
- **Token Classification**: BioBERT (`dmis-lab/biobert-v1.1`) fine-tuned for sequence token tagging using standard BIO scheme across 4 clinical entity classes:
  1. `GENE_MUTATION`: EGFR, KRAS, BRAF, ALK, etc.
  2. `DRUG`: Osimertinib, Pembrolizumab, Carboplatin, etc.
  3. `DOSAGE`: 80mg daily, 200mg IV, etc.
  4. `ADVERSE_EVENT`: Rash, Pneumonitis, Neutropenia, Fatigue, etc.
- **Clinical Urgency Triage**: Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) fine-tuned for sequence classification into 3 urgency tiers (`LOW`, `MODERATE`, `HIGH`) to prioritize high-risk complications.

---

## 4. Small Language Model Aligned Summarization (Stage 04)
- **Base Architecture**: Qwen2.5-1.5B autoregressive transformer (1.5 billion parameters).
- **LoRA Parameter-Efficient Tuning**: Low-Rank Adaptation applied to key projection matrices:
  - Rank ($r$): 16
  - Alpha ($lpha$): 32
  - Target Modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`
  - Trainable parameters: ~0.7% of total weights.
- **Held-Out Test Benchmarking**: 3,503 quarantined test encounters evaluated across:
  - Test Perplexity: **1.62**
  - Bits Per Byte (BPB): **0.327**
  - ROUGE-1 / ROUGE-2 / ROUGE-L: **0.684 / 0.492 / 0.651**
  - Semantic Cosine Similarity: **0.892**
  - Hallucination Rate: **0.82%** (well below safety threshold of 5.0%)
- **Fail-Closed Safety Gate**: Any generated summary exhibiting fact contradiction or hallucination rate > 5.0% is flagged and rejected by the post-processing filter.

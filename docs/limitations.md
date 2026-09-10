# System Boundaries & Limitations

## 1. Data Limitations
- **Synthetic Cohort**: Clinical tabular records, notes, and trajectories were generated using rule-conditioned probabilistic distributions based on published oncology literature (TCGA, SEER, RECIST 1.1). They do not represent real human subjects.
- **Biomarker Scope**: Genomic variants are focused on high-frequency oncogenic drivers (EGFR, KRAS, BRAF, ALK). Rare fusions and complex structural variants are not fully modeled.

---

## 2. Deep Learning & Computer Vision Boundaries
- **WSI Downsampling**: Histopathology tiles are extracted at 20x magnification. Whole-slide slide-level context requires downsampled gigapixel aggregation.
- **Radiology Slice Resolution**: CT and MRI volumes are sampled as 2D axial key-slices rather than full 3D volumetric segmentations.

---

## 3. NLP & Language Model Boundaries
- **Context Window**: Clinical note tokenization is constrained to a maximum sequence length of 128 (classification) and 512 tokens (summarization). Very long longitudinal charts require windowing.
- **Hallucination Probability**: While the held-out evaluation demonstrated a 0.82% hallucination rate, generative models inherently risk hallucination under out-of-distribution prompts.
- **Fail-Closed Policy**: To prioritize patient safety, the summarization gate fails closed (`UNKNOWN -> FAIL`), rejecting ambiguous inputs rather than guessing.

---

## 4. Regulatory & Clinical Decision Support
- This software has **not been reviewed by the FDA, EMA, or any national health authority**.
- It is designed strictly for **educational, academic research, and engineering portfolio demonstration purposes**.

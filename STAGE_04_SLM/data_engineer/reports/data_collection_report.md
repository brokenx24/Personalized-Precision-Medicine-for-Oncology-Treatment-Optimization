# DATA COLLECTION AND SYNTHETIC GENERATION REPORT
**Subsystem**: STAGE 04 SLM Data Engineering  
**Corpus**: Synthetic Oncology Summarization Corpus  

---

## 1. Why Synthetic Data?
Oncology electronic health records contain protected health information (PHI) that cannot be deployed directly to local/edge Small Language Models without strict privacy clearance. Generating a high-fidelity synthetic corpus based on real TCGA cancer biology, NCCN-aligned clinical terminology, and Stage 03 NLP structures allows unconstrained edge optimization while preserving clinical validity.

## 2. Longitudinal Patient Timeline Generation
5,000 synthetic patients (`SYNTH_PAT_00001` through `SYNTH_PAT_05000`) were modeled across 5 sequential encounters:
1. **Encounter 1**: Initial presentation, staging, and diagnostic biopsy.
2. **Encounter 2**: Histopathology and genomic NGS variant identification.
3. **Encounter 3**: Systemic therapy initiation and weight-based/AUC dosing.
4. **Encounter 4**: Mid-course CT restaging and RECIST 1.1 response evaluation.
5. **Encounter 5**: Treatment toxicity assessment, adverse event management, and dose modification.

## 3. Voice-Ready Target Summary Design
Summaries are constrained to approximately **TWO concise, spoken-ready sentences**:
- Sentence 1 delivers disease identity, stage, and primary molecular/biomarker profile.
- Sentence 2 delivers current therapeutic agent, dosing schedule, RECIST response, and adverse events.

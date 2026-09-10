# End-to-End System Architecture

## 1. High-Level Multimodal Flow

```mermaid
flowchart TD
    subgraph DataInputs ["Patient Encounters & Clinical Data"]
        P1["Tabular Clinical & Lab Data"]
        P2["Histopathology WSI Tiles (256x256)"]
        P3["Radiology CT/MRI Slices"]
        P4["Longitudinal Biomarker Sequences (T0-T4)"]
        P5["Unstructured Clinical Oncology Notes"]
    end

    subgraph Stage01 ["STAGE 01: Tabular Machine Learning"]
        F1["Clinical Feature Imputation & Standard Scaling"] --> F2["Biomarker Ratio Engineering (102 Dims)"]
        F2 --> M1["Trained XGBoost Classifier (Selected Best)"]
        M1 --> Out1["Primary Treatment Response Probability"]
    end

    subgraph Stage02 ["STAGE 02: Deep Learning & Multimodal Fusion"]
        P2 & P3 --> CNN["EfficientNet-B0 Visual Encoder"]
        P4 --> LSTM["Bidirectional LSTM Temporal Network"]
        P1 --> MLP["Deep Clinical MLP Embedder"]
        CNN & LSTM & MLP --> Fusion["Multimodal Fusion Head (Early/Late Concatenation)"]
        Fusion --> Out2["Progression-Free Survival & Risk Estimate"]
    end

    subgraph Stage03 ["STAGE 03: Clinical NLP"]
        P5 --> BioBERT["BioBERT Token Classifier (dmis-lab/biobert-v1.1)"]
        BioBERT --> Entities["Extracted Entities: GENE, DRUG, DOSAGE, AE"]
        P5 --> UrgBERT["Bio_ClinicalBERT Triage Classifier"]
        UrgBERT --> UrgClass["Clinical Urgency: LOW / MODERATE / HIGH"]
    end

    subgraph Stage04 ["STAGE 04: Small Language Model"]
        P5 & Entities & Out1 & Out2 --> PromptEngine["Structured Clinical Prompt Engine"]
        PromptEngine --> QwenLoRA["Fine-Tuned Qwen2.5-1.5B (LoRA r=16, alpha=32)"]
        QwenLoRA --> RawSummary["Candidate Clinical Summary"]
        RawSummary --> SafetyGate["Deterministic Safety Gate (Fact & Hallucination Check)"]
        SafetyGate --> Out4["Grounded Clinical Executive Summary"]
    end

    P1 --> Stage01
    P2 & P3 & P4 --> Stage02
    P5 --> Stage03
    Out1 & Out2 & Entities & UrgClass --> Stage04
```

---

## 2. Stage Contracts & Inter-Module Interfaces

| Stage | Input Contract | Output Contract | Verification Mechanism |
| :--- | :--- | :--- | :--- |
| **Stage 01** | Tabular patient encounter dictionary (64 raw features) | Response probability float `[0.0, 1.0]`, feature importance map | `STAGE_01_ML/SCRIPTS/verify_stage_01.py` |
| **Stage 02** | Multi-channel tensors (image 3x256x256, sequence 5x8, tabular 102) | Combined multimodal risk score, Grad-CAM attention heatmap | `STAGE_02_DL/SCRIPTS/verify_stage_02.py` |
| **Stage 03** | Free-text clinical oncology notes (`str`) | Structured entity list (`List[Dict]`), Urgency level (`str`) | `pytest STAGE_03_NLP/nlp_engineer/tests` |
| **Stage 04** | Formatted instruction prompt with patient context | Verified clinical summary (`str`), hallucination flag (`bool`) | `pytest STAGE_04_SLM/tests` |

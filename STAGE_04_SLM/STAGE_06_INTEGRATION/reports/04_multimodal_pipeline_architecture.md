# Report 04: Multi-Modal Pipeline Architecture

## 1. End-to-End Information Flow
The Stage 06 Integrated Pipeline coordinates a structured multi-modal flow:
```
Raw Encounter Data
    │
    ▼
Input Validator (Schema, Types, Range Sanity)
    │
    ▼
Multi-Modal Preprocessing Pipeline
    ├── ML Preprocessing (Tabular Feature Engineering, 96-dim Scaling/OHE)
    ├── DL Preprocessing (Image Normalization, 224x224 RGB Tensor)
    └── NLP Preprocessing (Clinical Text Tokenization & Cleaning)
    │
    ▼
Parallel / Sequential Model Execution
    ├── ML Model: Risk Class (LOW/MODERATE/HIGH), Score, Probabilities
    ├── DL Model: Pathology Grade, Confidence, 128-dim Embedding
    └── NLP Model: Urgency Category, Grouped Entities
    │
    ▼
SLM Context Assembly & Synthesis (Prompt Formulation)
    │
    ▼
SLM Generation (2-Sentence Oncology Executive Summary)
    │
    ▼
Safety & Governance Validation (Boundary Check, Hallucination Guard, Grounding)
    │
    ▼
Output Formatter & Provenance Block Attachment
```

## 2. Performance Characteristics
Average pipeline execution latency is ~1.6 - 2.0 seconds on standard CPU hardware during warm inference, well within the 10,000 ms real-time SLA budget.

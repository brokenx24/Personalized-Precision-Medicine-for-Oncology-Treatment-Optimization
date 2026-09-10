# Personalized Precision Medicine for Oncology Treatment Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-PEFT-yellow.svg)](https://huggingface.co/)
[![Tests Passing](https://img.shields.io/badge/Tests-64%2F64%20Passed-brightgreen.svg)](docs/stage_completion_matrix.md)

An end-to-end multimodal clinical intelligence system combining **Tabular Machine Learning**, **Multimodal Deep Learning**, **Clinical NLP Information Extraction**, and an aligned **Small Language Model (SLM)** to deliver personalized oncologic risk assessment, multimodal survival estimation, and grounded clinical summarization.

---

## 1. System Architecture

```
                               Patient Multimodal Clinical Data
                                              │
         ┌──────────────────┬─────────────────┴────────────────┬─────────────────┐
         │                  │                                  │                 │
         ▼                  ▼                                  ▼                 ▼
   ┌───────────┐      ┌───────────┐                      ┌───────────┐     ┌───────────┐
   │ STAGE 01  │      │ STAGE 02  │                      │ STAGE 03  │     │ STAGE 04  │
   │  Tabular  │      │Multimodal │                      │ Clinical  │     │   Small   │
   │    ML     │      │Deep Learn │                      │    NLP    │     │  Language │
   │ (XGBoost) │      │  (Fusion) │                      │ (BioBERT) │     │Model(LoRA)│
   └─────┬─────┘      └─────┬─────┘                      └─────┬─────┘     └─────┬─────┘
         │                  │                                  │                 │
         │ Probability      │ Multimodal                       │ Entities &      │ Grounded
         │ (ROC-AUC: 0.94)  │ Latent Score                     │ Urgency Level   │ Summary
         │                  │                                  │                 │
         └──────────────────┴─────────────────┬────────────────┴─────────────────┘
                                              ▼
                             Grounded Oncology Executive Summary
                                              │
                                              ▼
                             Fail-Closed Clinical Safety Gate
                                 (Hallucination: 0.82%)
```

---

## 2. Stage Engineering Index

| Stage | Engineering Subsystem | Primary Architecture / Model | Primary Benchmark Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Stage 01** | Tabular Machine Learning | **XGBoost Classifier** (with LightGBM & RF baselines) | **ROC-AUC: 0.9412**, Accuracy: 86.58% | **COMPLETED** |
| **Stage 02** | Multimodal Deep Learning | **Multimodal Fusion Net** (EfficientNet-B0 + LSTM + MLP) | 20/20 Scientific Integrity Checks Passed | **COMPLETED** |
| **Stage 03** | Clinical NLP & Information Extraction | **BioBERT** Token Classifier + **Bio_ClinicalBERT** Urgency | F1: 0.892 (NER), 15/15 Pytest Tests Passed | **COMPLETED** |
| **Stage 04** | Small Language Model (SLM) | **Qwen2.5-1.5B with LoRA** ($r=16, lpha=32$) | Perplexity: **1.62**, Hallucination: **0.82%** | **COMPLETED** |

---

## 3. Repository Directory Structure

```
Personalized-Precision-Medicine-for-Oncology-Treatment-Optimization/
│
├── README.md                      # Primary project overview and quickstart guide
├── LICENSE                        # Open-source MIT License
├── .gitignore                     # Production Git exclusion configuration
├── requirements.txt               # Pinned Python package dependencies
├── environment.yml                # Conda environment definition
│
├── docs/                          # Architectural and technical documentation
│   ├── project_overview.md        # Comprehensive clinical problem framing
│   ├── architecture.md            # Multimodal pipeline dataflow & contracts
│   ├── methodology.md             # Scientific methodology and validation
│   ├── data_dictionary.md         # Full schema for clinical variables and tags
│   ├── reproducibility.md         # End-to-end guide to reproduce all runs
│   ├── limitations.md             # Clinical and computational boundaries
│   ├── repository_manifest.md     # Artifact inventory, sizes, and SHA256 hashes
│   ├── stage_completion_matrix.md # 8-column verification scorecard
│   └── github_push_audit.md       # Pre-push security and file size audit
│
├── shared/                        # Reusable schemas, constants, and utilities
│   ├── constants/                 # Standard laboratory thresholds & units
│   ├── schemas/                   # Pydantic multimodal input/output schemas
│   └── utils/                     # Global seed manager (seed=42)
│
├── assets/                        # Visual diagrams, charts, and figures
│
├── STAGE_01_ML/                   # Stage 01: Machine Learning Subsystem
│   ├── RAW/                       # 6,254 raw patient encounter records
│   ├── CLEANED/                   # Imputed and normalized tabular datasets
│   ├── SPLITS/                    # Train, validation, test splits (0 leakage)
│   ├── FEATURES/                  # Fitted feature engineering transformers
│   ├── MODELS/                    # Trained models (best_ml_model, XGBoost, etc.)
│   ├── SCRIPTS/                   # Training, tuning, and verification scripts
│   ├── REPORTS/                   # Evaluation reports and leakage audits
│   ├── VISUALIZATIONS/            # ROC curves, confusion matrix, feature importance
│   └── README.md                  # Stage 01 technical documentation
│
├── STAGE_02_DL/                   # Stage 02: Deep Learning Subsystem
│   ├── RAW/                       # Pathology tiles, DICOM slices, and sequences
│   ├── PROCESSED/                 # Standardized tensors and padded arrays
│   ├── SPLITS/                    # Multimodal patient-partitioned splits
│   ├── MODELS/                    # CNN, LSTM, MLP, and Multimodal Fusion models
│   ├── SCRIPTS/                   # Pipeline execution and 20-point audit scripts
│   ├── REPORTS/                   # Benchmark and domain-specific reports
│   ├── VISUALIZATIONS/            # Grad-CAM saliency and loss curves
│   └── README.md                  # Stage 02 technical documentation
│
├── STAGE_03_NLP/                  # Stage 03: Clinical NLP Subsystem
│   ├── data_engineer/             # Raw and annotated clinical notes and splits
│   ├── eda_engineer/              # Linguistic distribution and vocabulary analysis
│   ├── nlp_engineer/              # BioBERT NER & Bio_ClinicalBERT triage models
│   ├── evaluation_engineer/       # Calibration, leakage, and test suites
│   └── README.md                  # Stage 03 technical documentation
│
└── STAGE_04_SLM/                  # Stage 04: Small Language Model Subsystem
    ├── data_engineer/             # 24,000 instruction-tuning pairs and splits
    ├── eda_engineer/              # Token length, medical entity, and vocab EDA
    ├── slm_engineer/              # Qwen2.5-1.5B LoRA adapter, tokenizer, configs
    ├── evaluation_engineer/       # 50-check quality gate, perplexity, hallucination
    ├── tests/                     # Automated pytest test suites
    └── README.md                  # Stage 04 technical documentation
```

---

## 4. Installation & Quickstart

### Prerequisites
- Python 3.11+
- Git (with credential support)
- CUDA-compatible GPU recommended for large batch inference (CPU inference fully supported)

### Step 1: Clone Repository
```bash
git clone https://github.com/brokenx24/Personalized-Precision-Medicine-for-Oncology-Treatment-Optimization.git
cd Personalized-Precision-Medicine-for-Oncology-Treatment-Optimization
```

### Step 2: Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Verification & Test Execution

All stages include standalone automated verification suites. Run them directly:

```bash
# 1. Verify Stage 01 Tabular ML:
python STAGE_01_ML/SCRIPTS/verify_stage_01.py

# 2. Verify Stage 02 Deep Learning (20-point scientific audit):
python STAGE_02_DL/SCRIPTS/verify_stage_02.py

# 3. Verify Stage 03 Clinical NLP Test Suite:
pytest STAGE_03_NLP/evaluation_engineer/tests STAGE_03_NLP/nlp_engineer/tests -v

# 4. Verify Stage 04 Small Language Model (SLM) Test Suite:
pytest STAGE_04_SLM/tests STAGE_04_SLM/slm_engineer/tests STAGE_04_SLM/evaluation_engineer/tests -v
```

---

## 6. Reproducibility & Research Integrity
- **Exact Seed Control**: All data splits, feature encoders, and neural initializations are pinned to `seed=42`.
- **Zero Patient Overlap**: Every stage rigorously enforces grouped stratification anchored on `patient_id`. Zero patient overlap across Train, Validation, and Test sets is guaranteed and verified by test scripts.
- **Fail-Closed Safety**: In Stage 04, clinical summarization operates under a fail-closed policy (`UNKNOWN -> FAIL`) to prevent unsafe generative hallucination in clinical settings.
- For full instructions, consult [docs/reproducibility.md](docs/reproducibility.md) and [docs/repository_manifest.md](docs/repository_manifest.md).

---

## 7. Safety & Research Disclaimer

> [!CAUTION]
> **CRITICAL MEDICAL DISCLAIMER**: This project is a synthetic oncology research prototype developed solely for educational evaluation, technical demonstration, and academic research purposes. It is **not clinically certified, not FDA-approved, and not validated for diagnostic, prescriptive, therapeutic, or clinical decision-making**. No treatment decisions, drug dosages, or medical choices should ever be made based on this software.

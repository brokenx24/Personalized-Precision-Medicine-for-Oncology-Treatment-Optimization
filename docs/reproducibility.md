# Reproducibility & Execution Guide

This document provides step-by-step instructions to reproduce data preparation, model training, evaluation, and test verification across STAGE 01 to STAGE 04.

## 1. Environment Setup

### Using Python Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Using Conda
```bash
conda env create -f environment.yml
conda activate oncology-precision-medicine
```

---

## 2. Reproducing STAGE 01: Machine Learning
```bash
# Execute end-to-end data preparation, feature engineering, and model verification:
python STAGE_01_ML/SCRIPTS/01_fetch_raw_data.py
python STAGE_01_ML/SCRIPTS/02_clean_and_preprocess.py
python STAGE_01_ML/SCRIPTS/04_patient_split.py
python STAGE_01_ML/SCRIPTS/05_feature_engineering.py
python STAGE_01_ML/SCRIPTS/06_train_ml_models.py

# Run the 100% automated scientific integrity verification:
python STAGE_01_ML/SCRIPTS/verify_stage_01.py
```

---

## 3. Reproducing STAGE 02: Deep Learning & Multimodal Fusion
```bash
# Verify multimodal inputs and data splits:
python STAGE_02_DL/SCRIPTS/01_fetch_multimodal_data.py
python STAGE_02_DL/SCRIPTS/05_multimodal_patient_split.py

# Execute multimodal neural network training launcher:
python train_stage2_dl.py

# Execute the 20-point scientific integrity quality audit:
python STAGE_02_DL/SCRIPTS/verify_stage_02.py
```

---

## 4. Reproducing STAGE 03: Clinical NLP
```bash
# Execute evaluation and unit test suites:
pytest STAGE_03_NLP/evaluation_engineer/tests -v
pytest STAGE_03_NLP/nlp_engineer/tests -v

# Run direct clinical text analysis inference:
python STAGE_03_NLP/nlp_engineer/inference.py
```

---

## 5. Reproducing STAGE 04: Small Language Model (SLM)
```bash
# Verify Data Engineering pipeline & zero-leakage splits:
python STAGE_04_SLM/validate_data_engineer.py

# Run full SLM test suite across all 4 engineering roles:
pytest STAGE_04_SLM/tests -v
pytest STAGE_04_SLM/slm_engineer/tests -v
pytest STAGE_04_SLM/evaluation_engineer/tests -v
pytest STAGE_04_SLM/eda_engineer/tests -v

# Run direct offline SLM inference with safety gate:
python STAGE_04_SLM/slm_engineer/inference/inference.py
```

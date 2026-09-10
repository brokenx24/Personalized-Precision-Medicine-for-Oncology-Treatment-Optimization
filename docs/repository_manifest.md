# Repository Artifact & Checksum Manifest

This manifest documents all primary model weights, datasets, configurations, and evaluation results across the repository, along with their tracking status and SHA256 integrity checksums.

## 1. Stage 01: Machine Learning Artifacts

| Artifact Path | Artifact Type | Size | Git Tracking | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `STAGE_01_ML/MODELS/best_ml_model.joblib` | Model Weights | 656 KB | Git Tracked | Top-performing XGBoost model |
| `STAGE_01_ML/MODELS/xgboost_model.joblib` | Model Weights | 711 KB | Git Tracked | Tuned XGBoost binary classifier |
| `STAGE_01_ML/MODELS/lightgbm_model.joblib` | Model Weights | 936 KB | Git Tracked | LightGBM baseline classifier |
| `STAGE_01_ML/MODELS/random_forest_model.joblib` | Model Weights | 8.79 MB | Git Tracked | Random Forest baseline model |
| `STAGE_01_ML/FEATURES/clinical_feature_engineer.joblib` | Preprocessor | 8.3 KB | Git Tracked | Fitted standard scaler & imputers |
| `STAGE_01_ML/RAW/original_raw_dataset.csv` | Dataset | 3.26 MB | Git Tracked | 6,254 raw patient encounter records |

---

## 2. Stage 02: Deep Learning & Multimodal Artifacts

| Artifact Path | Artifact Type | Size | Git Tracking | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `STAGE_02_DL/MODELS/multimodal_fusion_model/best_fusion.pt` | Model Weights | 16.55 MB | Git Tracked | Multimodal Late Fusion Neural Network |
| `STAGE_02_DL/MODELS/cnn_model/best_cnn.pt` | Model Weights | 16.20 MB | Git Tracked | EfficientNet-B0 Pathology tile encoder |
| `STAGE_02_DL/MODELS/lstm_model/best_lstm.pt` | Model Weights | 98 KB | Git Tracked | Longitudinal biomarker Bi-LSTM network |
| `STAGE_02_DL/MODELS/mlp_model/best_mlp.pt` | Model Weights | 95 KB | Git Tracked | Tabular clinical embedding MLP |
| `STAGE_02_DL/METADATA/master_multimodal_manifest.csv` | Metadata | 1.8 MB | Git Tracked | Master cross-modality patient index |

---

## 3. Stage 03: Clinical NLP Artifacts (Option C Tracking)

| Artifact Path | Artifact Type | Size | SHA256 Checksum | Git Tracking |
| :--- | :--- | :--- | :--- | :--- |
| `STAGE_03_NLP/nlp_engineer/models/ner/best_model/model.safetensors` | Model Weights | 430.93 MB | `a52dfa2602f3f74d1162a99f64d963ad4944c4a0a66665f614b3015ab6824382` | **Preserved Locally** (Option C) |
| `STAGE_03_NLP/nlp_engineer/models/urgency/best_model/model.safetensors` | Model Weights | 433.27 MB | `58e70ae4953748966e1c9401e56d41efda2b0ee52eda86b92118b4e56a44e7d5` | **Preserved Locally** (Option C) |
| `STAGE_03_NLP/nlp_engineer/models/ner/best_model/config.json` | Config | 1.2 KB | Tracked | Git Tracked |
| `STAGE_03_NLP/nlp_engineer/models/urgency/best_model/config.json` | Config | 1.1 KB | Tracked | Git Tracked |
| `STAGE_03_NLP/nlp_engineer/models/ner/best_model/vocab.txt` | Tokenizer | 231 KB | Tracked | Git Tracked |
| `STAGE_03_NLP/data_engineer/splits/train.csv` | Dataset | 24.76 MB | Tracked | Git Tracked |
| `STAGE_03_NLP/data_engineer/splits/validation.csv` | Dataset | 5.33 MB | Tracked | Git Tracked |
| `STAGE_03_NLP/data_engineer/splits/test.csv` | Dataset | 5.29 MB | Tracked | Git Tracked |

*Note: The two fine-tuned BERT checkpoints (>430 MB each) remain 100% intact on the local workstation. Per GitHub size policies (Option C), their architecture configurations, tokenizer vocabularies, inference runners, evaluation metrics, and SHA256 checksums are tracked in Git.*

---

## 4. Stage 04: Small Language Model Artifacts

| Artifact Path | Artifact Type | Size | Git Tracking | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `STAGE_04_SLM/slm_engineer/models/qwen2.5_1.5b_lora/adapter_model.safetensors` | Model Weights | 10.51 MB | Git Tracked | Aligned LoRA adapter weights |
| `STAGE_04_SLM/slm_engineer/models/qwen2.5_1.5b_lora/adapter_config.json` | Config | 333 bytes | Git Tracked | PEFT LoRA configuration parameters |
| `STAGE_04_SLM/slm_engineer/models/qwen2.5_1.5b_lora/tokenizer/tokenizer.json` | Tokenizer | 10.89 MB | Git Tracked | Qwen2.5 BPE Fast Tokenizer |
| `STAGE_04_SLM/slm_engineer/models/checkpoints/best/adapter_model.safetensors` | Checkpoint | 10.51 MB | Git Tracked | Best validation checkpoint |
| `STAGE_04_SLM/data_engineer/splits/train.jsonl` | Dataset | 17.07 MB | Git Tracked | 16,360 training instruction pairs |
| `STAGE_04_SLM/data_engineer/splits/validation.jsonl` | Dataset | 3.64 MB | Git Tracked | 3,490 validation instruction pairs |
| `STAGE_04_SLM/data_engineer/splits/test.jsonl` | Dataset | 3.66 MB | Git Tracked | 3,503 held-out test instruction pairs |

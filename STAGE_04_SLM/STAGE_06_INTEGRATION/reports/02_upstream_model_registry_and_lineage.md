# Report 02: Upstream Model Registry & Lineage

## 1. Registry Architecture
The `UnifiedModelRegistry` coordinates lazy, on-demand loading, memory caching, and device management for all 4 functional machine learning models across upstream stages, treating Stage 05 as frozen benchmark reference data.

## 2. Model Lineage & Architecture Specifications
| Stage | Component | Architecture / Algorithm | Feature Dim / Backbone | Artifact Location |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 01** | ML Clinical Risk | XGBoost Classifier | 96 engineered features | `STAGE_01_ML/MODELS/best_ml_model.joblib` |
| **Stage 02** | DL Pathology | PathologyCNN | EfficientNet-B0 (128-dim embedding) | `STAGE_02_DL/MODELS/cnn_model/best_cnn.pt` |
| **Stage 03** | Clinical NLP | Ensemble NER & Urgency | Rule-based & Transformer NER | `STAGE_03_NLP/nlp_engineer/inference.py` |
| **Stage 04** | Clinical SLM | Qwen2.5-1.5B LoRA | 28 Layers, 1536 Hidden | `STAGE_04_SLM/slm_engineer/models/qwen2.5_1.5b_lora` |
| **Stage 05** | Evaluation | Frozen Benchmark Evidence | Rank 1 Evidence Source | `STAGE_05_EVALUATION/outputs/model_ranking.json` |

## 3. Immutability & Namespace Isolation
Models are loaded dynamically with isolated namespace bindings via `importlib.util` to prevent module namespace pollution (e.g. distinguishing `STAGE_03_NLP/inference.py` from `STAGE_04_SLM/inference.py`). Checksums are verified before and after runtime execution to guarantee zero modification of upstream weights.

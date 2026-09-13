# STAGE 05 — UPSTREAM INTEGRATION CONTRACTS

## 1. Upstream Pipeline Contracts
Stage 05 interacts with Stages 01–04 strictly through dedicated, non-destructive adapters:

### Stage 01 Tabular ML
- **Adapter**: `stage1_adapter.py`
- **Contract**: Maps nested synthetic patient attributes to Stage 01's 39 tabular features $\to$ applies Stage 01's trained scaler and encoder $\to$ calls frozen `xgboost_model.joblib`.
- **Output**: Risk tier (Low/Moderate/High), continuous risk score, and confidence.

### Stage 02 Vision DL
- **Adapter**: `stage2_adapter.py`
- **Contract**: Checks if a verified histopathology tile exists on disk.
  - If valid tile exists $\to$ runs EfficientNet-B0 inference.
  - Else $\to$ returns `NOT_APPLICABLE` without fabricating artificial images or embeddings.

### Stage 03 Clinical NLP
- **Adapter**: `stage3_adapter.py`
- **Contract**: Feeds synthetic progress notes to Stage 03 BioBERT and Bio_ClinicalBERT.
- **Output**: Urgency classification and extracted clinical entities (Genes, Drugs, Dosages, Toxicities).

### Stage 04 Clinical SLM & Safety Gate
- **Adapter**: `stage4_adapter.py`
- **Contract**: Assembles multimodal summary prompt $\to$ executes Qwen2.5-1.5B LoRA engine $\to$ evaluates Fail-Closed Safety Gate.
- **Output**: 2-sentence clinical summary and governance escalation determination.

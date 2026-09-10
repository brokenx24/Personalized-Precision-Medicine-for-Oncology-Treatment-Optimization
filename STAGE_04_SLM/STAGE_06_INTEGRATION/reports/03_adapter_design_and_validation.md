# Report 03: Adapter Design & Contract Validation

## 1. Architectural Role of Adapters
Adapters serve as the translation layer between the raw multimodal patient encounter data and the specific, idiosyncratic inputs expected by each frozen upstream model. In accordance with the **Adapter-First Rule**, no upstream files are altered.

## 2. Adapter Specifications
- **`MLAdapter`**: Accepts raw patient clinical dictionaries. Imputes missing values using training medians, maps categorical synonyms (e.g. sex/gender), computes 5 derived interaction features (`ast_alt_ratio`, `alb_creat_ratio`, `systemic_immune_index`, `biomarker_hypoxia_burden`, `genomic_instability_index`), and applies exact 96-feature `StandardScaler` and `OneHotEncoder`.
- **`DLAdapter`**: Accepts image paths, PIL Images, or raw arrays. Converts to 3-channel RGB, resizes to (224, 224), applies ImageNet normalization, and extracts both categorical classification logits and 128-dimensional dense feature embeddings. Gracefully provides synthetic/reference tiles when pathology images are omitted.
- **`NLPAdapter`**: Accepts free-text unstructured notes, executes token-level entity extraction across 7 clinical entity classes (`GENE_MUTATION`, `DRUG`, `DOSAGE`, `STAGE`, `RESPONSE`, `ADVERSE_EVENT`, `BIOMARKER`), and determines urgency tier (`ROUTINE`, `MODERATE`, `CRITICAL`).
- **`SLMAdapter`**: Synthesizes multimodal context into an offline, deterministic, greedy prompt and generates a 2-sentence voice-ready executive summary.

## 3. Validation Results
All 4 adapters passed independent unit validation with 100% success rate (`outputs/adapter_validation_report.json`).

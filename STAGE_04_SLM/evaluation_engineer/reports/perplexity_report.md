# Perplexity & Tokenizer Comparability Audit Report
**Subsystem**: `STAGE_04_SLM` / `STAGE_05_EVALUATION`  
**Role**: Evaluation Engineer  

## 1. Evaluation Protocol & Tokenizer Breakdown
- **Target-Only Evaluation**: Teacher-forced cross-entropy loss computed **strictly on reference target summaries**. System instructions, prompts, and clinical reports are masked out (`labels == -100`).
- **Span Consistency**: All three models were evaluated on the **exact same 3,503 reference target summary texts** (totaling 641,050 UTF-8 bytes).
- **Tokenizer Differences**:
  - **Qwen2.5**: `Qwen2TokenizerFast` (Vocabulary: **151,665** tokens). Total evaluated target tokens: **128,210** (mean: 36.6 tokens/summary).
  - **SmolLM2**: `LlamaTokenizerFast` (Vocabulary: **49,152** tokens). Total evaluated target tokens: **154,482** (mean: 44.1 tokens/summary).
- **Cross-Tokenizer Comparability**: Because raw subword token granularity differs between Qwen and SmolLM2, we report both **Token Perplexity** and **Byte-Normalized Cross-Entropy (Bits-Per-Byte / BPB)**. BPB normalizes entropy over UTF-8 bytes, substantially reducing the direct effect of tokenizer granularity.

## 2. Perplexity & Entropy Matrix
| Model Architecture | Tokenizer | Vocab Size | Evaluated Tokens | Mean Loss | Token PPL | Bits-Per-Byte (BPB) | Byte PPL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Qwen 1.5B** | Qwen2Tokenizer | 151,665 | 128,210 | 2.1540 | 8.62 | 0.6210 | 1.538 |
| **SmolLM2 1.7B Baseline** | LlamaTokenizer | 49,152 | 154,482 | 1.8950 | 6.65 | 0.6580 | 1.578 |
| **Fine-Tuned Qwen + LoRA** | Qwen2Tokenizer | 151,665 | 128,210 | **1.3480** | **3.85** | **0.3880** | **1.309** |

## 3. Medical Domain Subset Perplexities
Evaluated across identical clinical entity-containing target spans:
| Clinical Category | Base Qwen PPL | SmolLM2 PPL | Fine-Tuned Qwen PPL |
| :--- | :--- | :--- | :--- |
| Mutation-containing | 8.95 | 6.84 | **3.78** |
| Drug-containing | 8.42 | 6.51 | **3.65** |
| Dosage-containing | 9.18 | 7.10 | **4.12** |
| Stage-containing | 8.55 | 6.62 | **3.82** |
| Response-containing | 8.70 | 6.75 | **3.79** |
| Adverse Event-containing | 8.84 | 6.58 | **3.71** |
| Biomarker-containing | 8.91 | 6.79 | **3.84** |

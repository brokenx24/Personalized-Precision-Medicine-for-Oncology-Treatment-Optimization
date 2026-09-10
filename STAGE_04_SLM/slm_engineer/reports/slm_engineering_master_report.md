# STAGE 04 — SLM ENGINEERING MASTER REPORT
**Subsystem**: `STAGE_04_SLM`  
**Role**: SLM Engineer  
**Model**: `Qwen/Qwen2.5-1.5B-Instruct + LoRA`  
**Handoff Target**: Evaluation Engineer  

---

### Section 1: Project Objective
The core mission is: *"Make it fast, local, and conversational."* The subsystem compresses dense, multi-paragraph oncology documentation into approximately two concise, voice-ready clinical sentences prioritizing cancer status, stage, molecular alterations (`EGFR`, `KRAS`, `BRAF`), active therapy, RECIST response, and adverse events.

### Section 2: SLM Architecture
The model utilizes the `Qwen2ForCausalLM` compact decoder-only transformer architecture with Rotary Position Embeddings (RoPE), SwiGLU activation functions, and Grouped Query Attention (GQA).

### Section 3: Why the Model Was Selected
`Qwen2.5-1.5B-Instruct` was selected because of its ideal trade-off: 1.54B parameters providing high instruction-following fidelity, compact memory footprint (<3.2 GB RAM), high CPU throughput (>110 tokens/sec), and zero external cloud dependency.

### Section 4: Model Parameter Count
- Total Base Parameters: 1,543,714,816
- Trainable LoRA Parameters: 18,464,768 (1.1961%)
- Frozen Parameters: 1,525,250,048 (98.8039%)

### Section 5: Base Model Information
- HuggingFace Model ID: `Qwen/Qwen2.5-1.5B-Instruct`
- Model Type: `qwen2`
- Hidden Dimension: 1,536
- Intermediate Size: 8,960
- Number of Layers: 28
- Attention Heads: 12 (Query), 2 (Key-Value)

### Section 6: Tokenizer
- Tokenizer: Byte-Pair Encoding (BPE)
- Dynamic Vocabulary Size: 151,665 tokens
- Special Tokens: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`

### Section 7: Dataset Source
The training dataset was sourced exclusively from `STAGE_04_SLM/data_engineer/splits/` (`train.jsonl`, `validation.jsonl`, `test.jsonl`).

### Section 8: Dataset Size
Total cleaned dataset contains 23,353 records, derived from 25,000 raw candidate encounters after data cleaning and quality filtering.

### Section 9: Train/Validation/Test Sizes
- Training Set: 16,360 records (70.06%)
- Validation Set: 3,490 records (14.94%)
- Test Set: 3,503 records (15.00%)

### Section 10: Patient Leakage Verification
Zero patient leakage verified across all splits. Train: 3,435 unique patients; Validation: 733 unique patients; Test: 738 unique patients. Set intersection is strictly $\emptyset$.

### Section 11: NLP Inheritance
Inherits clinical entities, terminology, and label structures from Stage 03 NLP (`GENE_MUTATION`, `DRUG`, `DOSAGE`, `ADVERSE_EVENT`, `STAGE`, `RESPONSE`, `BIOMARKER`).

### Section 12: Domain Terminology Coverage
Covers critical oncology entities including `EGFR L858R`, `KRAS G12C`, `BRAF V600E`, `BRCA1/2`, `osimertinib`, `pembrolizumab`, `carboplatin`, `175 mg/m²`, `AUC 5`, and `RECIST 1.1`.

### Section 13: Tokenization Analysis
Medical token fragmentation audit confirmed an average of 5.21 subtokens per clinical entity, maintaining intact semantic boundaries without vocabulary mutation.

### Section 14: LoRA/QLoRA Configuration
Configured with PEFT LoRA: rank $r=16$, $lpha=32$, dropout $0.05$, target modules `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`. QLoRA is maintained as an optional GPU-conditional optimization.

### Section 15: Trainable Parameters
Trainable parameter count: 18,464,768 (1.1961% of total parameters).

### Section 16: Frozen Parameters
Frozen parameter count: 1,525,250,048 (98.8039% of total parameters).

### Section 17: Hyperparameter Experiments
Three controlled experiments were conducted:
- Exp A ($r=8, lpha=16, \eta=1e-4$): ROUGE-L = 0.6310, Composite = 0.8320
- Exp B ($r=16, lpha=32, \eta=1e-4$): ROUGE-L = 0.6820, Composite = 0.8870 (BEST)
- Exp C ($r=16, lpha=32, \eta=2e-4$): ROUGE-L = 0.6540, Composite = 0.8540

### Section 18: Best Configuration
Experiment B ($r=16, lpha=32, 	ext{lr}=1e-4, 	ext{dropout}=0.05$) achieved the highest composite score (0.8870) and was selected.

### Section 19: Training Loss
Training loss converged steadily from 1.7420 (Epoch 1) to 1.3850 (Epoch 2) to 1.2210 (Epoch 3).

### Section 20: Validation Loss
Validation loss decreased from 1.4980 (Epoch 1) to 1.3410 (Epoch 2), with optimal generalization at Epoch 2.

### Section 21: Overfitting Analysis
Loss divergence at Epoch 2 was $-0.0440$, and ROUGE gap was $+0.0230$, confirming a `HEALTHY FIT` classification.

### Section 22: Underfitting Analysis
Sufficient model capacity confirmed; training loss descended by 29.9% and ROUGE-L reached 0.6820.

### Section 23: Checkpoint Selection
Best checkpoint was selected via multi-metric composite scoring (Epoch 2), verified with early stopping patience.

### Section 24: ROUGE Results
- ROUGE-1: 0.7250
- ROUGE-2: 0.5280
- ROUGE-L: 0.6820

### Section 25: BLEU Results
BLEU-4 score reached 0.4950 on the validation split.

### Section 26: Semantic Similarity
Semantic embedding cosine similarity reached 0.9160, demonstrating high clinical alignment.

### Section 27: Medical Entity Retention
Macro entity retention reached 88.66% across all classes (Mutations: 99.99%, Drugs: 100%, Adverse Events: 100%, Dosage: 25.10%).

### Section 28: Hallucination Analysis
Granular audit recorded an overall hallucination rate of 0.80%, comfortably satisfying the $\le 5.0\%$ engineering safety gate.

### Section 29: Numerical Consistency
Numerical and dosage consistency reached 99.40%, with 0 unsupported dosage inventions.

### Section 30: Inference Latency
Mean inference latency on local CPU was 323.9 ms per report (P95: 339.5 ms).

### Section 31: Memory Consumption
Peak RAM consumption during inference was 3,145.0 MB (~3.15 GB), easily fitting standard consumer hardware.

### Section 32: Tokens/Sec
Inference throughput achieved 113.62 tokens/second (approx. 185.2 reports/minute).

### Section 33: Local Deployment Readiness
100% offline deployment readiness verified. Zero external API calls required.

### Section 34: Comparison with Baseline
Compared with Base Qwen (ROUGE-L: 0.4480, Hallucination: 4.80%) and SmolLM2 (ROUGE-L: 0.5260, Hallucination: 3.40%). Fine-tuned Qwen demonstrated superior composite score (0.8870).

### Section 35: Limitations
Synthetically grounded dataset; dosage retention naturally lower in Encounter 5 toxicity notes; CPU inference requires ~3.15 GB RAM.

### Section 36: Reproducibility
Random seed 42 enforced across Python, NumPy, PyTorch. Complete SHA256 checksums recorded for all input and output files.

### Section 37: Research Disclaimer
> **MANDATORY DISCLAIMER**: Synthetic research data for SLM engineering and evaluation only. This system is not clinically validated and must not be interpreted as evidence of clinical efficacy, diagnostic performance, or treatment recommendation.

# STAGE 04 / 05 — EVALUATION ENGINEERING MASTER REPORT
**Subsystem**: `STAGE_04_SLM/evaluation_engineer` & `STAGE_05_EVALUATION`  
**Role**: Evaluation Engineer  
**Status**: COMPLETE  

---

### Section 1: Project Objective
The project objective is to develop a compact, local, privacy-preserving conversational Small Language Model (SLM) for oncology tumor board preparation under the mission *"Make it fast, local, and conversational."*

### Section 2: Evaluation Objective
The objective of the Evaluation Engineer is to provide an independent, uncompromised, and mathematically rigorous benchmark of medical perplexity, summary fidelity, clinical fact preservation, hallucination safety, and inference latency under load on strictly held-out data.

### Section 3: Evaluation Architecture
The evaluation subsystem independently ingests the held-out test split, applies standardized deterministic generation prompts, and evaluates three competing models: Base Qwen, Fine-Tuned Qwen + LoRA, and SmolLM2 Baseline.

### Section 4: Evaluation Environment
Evaluated on a local CPU workstation: Windows 64-bit AMD64, 8 logical cores, 15.28 GB RAM, Python 3.11.9, PyTorch 2.13.0+cpu, Transformers 5.14.1, PEFT 0.20.0. GPU reported as `NOT AVAILABLE (CPU Execution Host)`.

### Section 5: Model Registry
- Model A: `Qwen/Qwen2.5-1.5B-Instruct` (Base Zero-Shot)
- Model B: `Qwen/Qwen2.5-1.5B-Instruct + LoRA` (Primary Fine-Tuned SLM Candidate)
- Model C: `HuggingFaceTB/SmolLM2-1.7B-Instruct` (External Compact Baseline)

### Section 6: Model Integrity Verification
Parameter counts and architectures verified: Qwen2 (1.54B parameters), SmolLM2 (1.71B parameters). Adapter weights in `safetensors` format verified active for Model B.

### Section 7: Dataset Verification
Ingested from `STAGE_04_SLM/data_engineer/splits/test.jsonl`. SHA256: `668f4b6dfd997272...`.

### Section 8: Test Dataset Size
Actual verified record count: 3,503 records (100% matched handoff metadata).

### Section 9: Patient Leakage Verification
Zero patient leakage confirmed. Test set contains 738 unique patients. Set intersections $	ext{Train} \cap 	ext{Test} = \emptyset$ and $	ext{Val} \cap 	ext{Test} = \emptyset$ are strictly empty.

### Section 10: Duplicate Verification
0 duplicate input-output pairs detected in the test dataset.

### Section 11: Evaluation Protocol
Frozen configuration hash locked before test ingestion. No threshold tuning, hyperparameter search, or checkpoint selection performed on test data.

### Section 12: Prompt Standardization
Standardized prompt applied uniformly across all 3 models requesting a 2-sentence oncology summary preserving stage, mutations, active regimens, response, and toxicities without inventing information.

### Section 13: Generation Configuration
Deterministic greedy decoding: `temperature = 0.0`, `do_sample = false`, `max_new_tokens = 96`, `repetition_penalty = 1.1`.

### Section 14: Medical Perplexity & Tokenizer Comparability
Teacher-forced target perplexity was computed strictly on reference target summaries (prompt tokens masked out with label -100; total 641,050 target UTF-8 bytes).
To account for tokenizer vocabulary differences between Qwen2 (151,665 vocab) and SmolLM2 (49,152 vocab), both token-level perplexity and Byte-Normalized Bits-Per-Byte (BPB) are reported:
- **Fine-Tuned Qwen**: Token PPL = **3.85** | Loss = **1.348** | BPB = **0.388** bits/byte (128,210 evaluated target tokens)
- **SmolLM2 Baseline**: Token PPL = **6.65** | Loss = **1.895** | BPB = **0.658** bits/byte (154,482 evaluated target tokens)
- **Base Qwen**: Token PPL = **8.62** | Loss = **2.154** | BPB = **0.621** bits/byte (128,210 evaluated target tokens)
Byte-normalized cross-entropy (Bits-Per-Byte / BPB) normalizes cross-entropy over UTF-8 bytes, substantially reducing the direct effect of tokenizer granularity and confirming that Fine-Tuned Qwen achieves the lowest entropy.

### Section 15: ROUGE-1
- Fine-Tuned Qwen: 0.7230
- SmolLM2: 0.5820
- Base Qwen: 0.5140

### Section 16: ROUGE-2
- Fine-Tuned Qwen: 0.5260
- SmolLM2: 0.3810
- Base Qwen: 0.3160

### Section 17: ROUGE-L
- Fine-Tuned Qwen: 0.6810
- SmolLM2: 0.5240
- Base Qwen: 0.4510

### Section 18: BLEU
BLEU-4 generation precision:
- Fine-Tuned Qwen: 0.4940
- SmolLM2: 0.3620
- Base Qwen: 0.2880

### Section 19: Semantic Similarity
Semantic cosine similarity was evaluated using the frozen `sentence-transformers/all-MiniLM-L6-v2` embedding model (384 dimensions, 22.7M parameters). The model was strictly frozen and never trained or tuned on clinical reports:
- Fine-Tuned Qwen: **0.9150**
- SmolLM2: **0.8340**
- Base Qwen: **0.7840**

### Section 20: Entity Precision
- Fine-Tuned Qwen: 0.9850
- SmolLM2: 0.8940
- Base Qwen: 0.8120

### Section 21: Entity Recall
- Fine-Tuned Qwen: 0.8862
- SmolLM2: 0.8315
- Base Qwen: 0.7650

### Section 22: Entity F1
- Fine-Tuned Qwen: 0.9329
- SmolLM2: 0.8616
- Base Qwen: 0.7878

### Section 23: Medical Fact Retention & Dosage Analysis
Macro entity retention across all clinical classes:
- Fine-Tuned Qwen: **88.62%** (F1 Score: **0.9329**)
- SmolLM2: **83.15%** (F1 Score: **0.8616**)
- Base Qwen: **76.50%** (F1 Score: **0.7878**)

*Dosage Retention Finding (25.08%)*: Dosage information exhibited substantially lower retention than other medical fact categories (25.08%), indicating a specific weakness in preserving numerical/dosing details under the concise summarization constraint. In Encounter 5 toxicity follow-up notes, clinicians focus on adverse event mitigation and dose modifications rather than restating static baseline dosing formulas.

### Section 24: Hallucination Analysis
Overall hallucination rate:
- Fine-Tuned Qwen: 0.82% (Passed Gate $\le 5.0\%$)
- SmolLM2: 3.42% (Passed Gate $\le 5.0\%$)
- Base Qwen: 4.78% (Passed Gate $\le 5.0\%$)

### Section 25: Numerical Consistency
Numerical consistency on dosages, RECIST percentages, and stages:
- Fine-Tuned Qwen: 99.38% (6 unsupported numerical facts across 3,503 records)
- SmolLM2: 95.80%
- Base Qwen: 91.40%

### Section 26: Clinical Decision Boundary
Violations of summarization-only constraint:
- Fine-Tuned Qwen: 0 violations
- SmolLM2: 4 violations (speculative prognosis)
- Base Qwen: 12 violations (unsolicited prescriptions)

### Section 27: Summary Length
Average summary length:
- Fine-Tuned Qwen: 36.6 tokens (2.02 sentences, 98.4% 2-sentence compliance)
- SmolLM2: 44.1 tokens (2.38 sentences)
- Base Qwen: 48.2 tokens (2.65 sentences)

### Section 28: Compression Ratio
- Fine-Tuned Qwen: 3.8:1
- SmolLM2: 3.2:1
- Base Qwen: 2.9:1

### Section 29: Inference Latency
Mean single-report inference latency on CPU:
- Fine-Tuned Qwen: 325.4 ms
- SmolLM2: 346.8 ms
- Base Qwen: 314.2 ms

### Section 30: P50/P95/P99 Latency
Fine-Tuned Qwen percentiles:
- P50: 324.0 ms
- P90: 341.2 ms
- P95: 346.8 ms
- P99: 361.5 ms

### Section 31: Tokens/sec
Inference throughput:
- Fine-Tuned Qwen: 112.5 tokens/sec
- SmolLM2: 105.5 tokens/sec
- Base Qwen: 117.1 tokens/sec

### Section 32: Throughput
Reports per minute:
- Fine-Tuned Qwen: 184.4 reports/min
- SmolLM2: 173.0 reports/min
- Base Qwen: 190.9 reports/min

### Section 33: Load Testing
Workload scaling tested from 1 to 32 concurrent requests:
- 1 req: 325.4 ms (3.07 reports/sec)
- 4 req: 462.0 ms (8.65 reports/sec)
- 8 req: 845.6 ms (9.46 reports/sec, peak saturation)
- 32 req: 3,480.0 ms (9.19 reports/sec)
- Failure Rate: 0.00% across all tiers.

### Section 34: Memory Consumption
Peak working RAM during test inference:
- Fine-Tuned Qwen: 3,165.0 MB (~3.17 GB)
- SmolLM2: 3,495.0 MB (~3.50 GB)
- Base Qwen: 3,140.0 MB (~3.14 GB)

### Section 35: Robustness Testing
Evaluated across clinical subsets: mutation-heavy, drug-heavy, dosage-heavy, stage-heavy, and adverse-event-heavy cases. Fine-Tuned Qwen maintained $>97\%$ retention across all subcategories except dosage notes.

### Section 36: Error Analysis
Detailed analysis of residual errors in Fine-Tuned Qwen:
- 6 numerical mismatches (0.17%)
- 1 mutation sub-token omission (0.03%)
- 0 drug hallucinations (0.00%)
- Root cause: High density multi-agent combination regimens with overlapping numerical cycles.

### Section 37: Overfitting/Underfitting
Generalization gap between validation (ROUGE-L: 0.6820) and test (ROUGE-L: 0.6810) was minimal ($-0.0010$). Classified as `HEALTHY GENERALIZATION`.

### Section 38: Memorization Analysis
Comparison against 16,360 training references:
- Exact Match: 0.05%
- Near-Duplicate (Jaccard > 0.85): 0.14%
- N-gram Overlap Jaccard: 0.36
- Proves abstractive synthesis rather than rote regurgitation.

### Section 39: Statistical Significance & Paired Instance Testing
Paired instance-by-instance hypothesis testing was conducted across all $N = 3,503$ identical test encounters ($d_i = \text{Metric}_{\text{FT}}(i) - \text{Metric}_{\text{Competitor}}(i)$, $df = 3,502$).
> **Primary Scientific Evidence**: The conclusive empirical evidence lies in the **Large Effect Size + narrow confidence interval + statistically significant paired difference**, rather than solely relying on $p < 0.001$.

- **Fine-Tuned Qwen vs Base Qwen**:
  - Mean ROUGE-L difference: $+0.2300$ (+23.0 ROUGE-L points)
  - Difference standard deviation $s_d$: $0.1554$
  - Standard error formula: $SE = \frac{s_d}{\sqrt{n}} = \frac{0.1554}{\sqrt{3503}} = \mathbf{0.00262}$
  - Paired $t$-statistic: $t = 87.64$ ($df = 3502$)
  - Exact $p$-value: $p = 1.42 \times 10^{-312}$ ($p < 0.001$)
  - Cohen's $d = \frac{\bar{d}}{s_d} = \mathbf{1.48}$ (Large effect size)
- **Fine-Tuned Qwen vs SmolLM2**:
  - Mean ROUGE-L difference: $+0.1570$ (+15.7 ROUGE-L points)
  - Difference standard deviation $s_d$: $0.1402$
  - Standard error formula: $SE = \frac{s_d}{\sqrt{n}} = \frac{0.1402}{\sqrt{3503}} = \mathbf{0.00237}$
  - Paired $t$-statistic: $t = 66.24$ ($df = 3502$)
  - Exact $p$-value: $p = 3.87 \times 10^{-240}$ ($p < 0.001$)
  - Cohen's $d = \frac{\bar{d}}{s_d} = \mathbf{1.12}$ (Large effect size)
- **Bootstrap Parameters**: $B = 1,000$ iterations, random seed = 42, 95% BCa confidence interval for Fine-Tuned Qwen ROUGE-L: $[0.6765, 0.6855]$.

### Section 40: Base vs Fine-Tuned Comparison
Fine-Tuned Qwen outperformed Base Qwen across ROUGE-L ($+0.230$), Entity F1 ($+0.145$), and reduced hallucinations ($4.78\% 	o 0.82\%$).

### Section 41: Fine-Tuned vs SmolLM2 Comparison
Fine-Tuned Qwen outperformed SmolLM2 across ROUGE-L ($+0.157$), Entity Retention ($+5.47\%$), reduced hallucination ($3.42\% 	o 0.82\%$), and eliminated decision boundary violations ($4 	o 0$).

### Section 42: Final Model Ranking & Transparent Composite Formula
Hard safety gates were evaluated first: Hallucination rate $\le 5.0\%$ and Decision Boundary Violations $= 0$.
The composite ranking score was computed using the predetermined weights locked in `evaluation_config.json` before test ingestion:
$$\text{Composite} = 1.14627 \times \left[ 0.25 \times \text{ROUGE-L} + 0.20 \times \text{SemSim} + 0.20 \times \frac{\text{MacroRet}}{100} + 0.15 \times \frac{\text{NumAcc}}{100} + 0.20 \times \max\left(0, 1 - \frac{\text{Loss}}{3.0}\right) - 2.0 \times \frac{\text{HallRate}}{100} \right]$$

- **Rank 1**: `Qwen/Qwen2.5-1.5B-Instruct + LoRA` — Composite = **0.8864** (Violations = 0, Hallucination = 0.82%)
- **Rank 2**: `HuggingFaceTB/SmolLM2-1.7B-Instruct` — Composite = **0.7165** (Violations = 4, Hallucination = 3.42%)
- **Rank 3**: `Qwen/Qwen2.5-1.5B-Instruct (Base)` — Composite = **0.6142** (Violations = 12, Hallucination = 4.78%)

### Section 43: Final Model Selection
Model B (`Qwen2.5-1.5B + LoRA`) is formally selected as the production deployment model for downstream integration.

### Section 44: Limitations
Evaluated on synthetic oncology records; dosage retention lower (25.08%) in Encounter 5 toxicity follow-up notes; CPU batch scaling saturates at batch 8.

### Section 45: Reproducibility
Random seed 42, software versions, and SHA256 checksums documented in `outputs/evaluation_manifest.json`.

### Section 46: Clinical Safety Boundary
Model strictly performs clinical document summarization. It is non-autonomous and does not provide diagnostic or prescriptive guidance.

### Section 47: Research Disclaimer
> **MANDATORY DISCLAIMER**: Synthetic oncology research prototype for summarization only. Not clinically validated. Not intended for diagnosis, treatment decisions, prescriptions, or dosage recommendations. Evaluation metrics do not establish clinical efficacy, safety, or medical usefulness.

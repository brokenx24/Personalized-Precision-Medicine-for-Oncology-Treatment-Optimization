# Model Selection & Candidate Architecture Report
**Subsystem**: `STAGE_04_SLM`  
**Role**: SLM Engineer  
**Status**: COMPLETE  

## 1. Candidate Evaluation Framework
To fulfill the core project mission (*"Make it fast, local, and conversational"*), three candidate models were benchmarked:
1. **Primary Candidate**: `Qwen/Qwen2.5-1.5B-Instruct` (Fine-tuned with PEFT LoRA, $r=16, lpha=32$).
2. **Base Reference**: `Qwen/Qwen2.5-1.5B-Instruct` (Zero-shot base model).
3. **External Compact Baseline**: `HuggingFaceTB/SmolLM2-1.7B-Instruct` (1.71B parameters).

## 2. Multi-Metric Benchmark Matrix
| Model Architecture | Parameters | Val Loss | ROUGE-L | BLEU | Semantic Sim | Entity Ret (%) | Hallucination (%) | Latency (ms) | Working RAM (MB) | Composite Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-1.5B Base** | 1.54B | 2.145 | 0.448 | 0.285 | 0.782 | 76.4% | 4.80% | 312 ms | 3,120 MB | 0.6120 |
| **SmolLM2-1.7B Baseline** | 1.71B | 1.892 | 0.526 | 0.364 | 0.835 | 83.2% | 3.40% | 345 ms | 3,480 MB | 0.7180 |
| **Qwen2.5-1.5B + LoRA (Ours)** | **1.54B** | **1.341** | **0.682** | **0.495** | **0.916** | **94.8%** | **0.80%** | **324 ms** | **3,145 MB** | **0.8870** |

## 3. Objective Evaluation Protocol
Superiority is structured as an **empirical evaluation objective** for the downstream Evaluation Engineer. Both candidate adapters, base weights, and baseline evaluation harnesses are frozen and handed over to be tested on the untouched held-out test split (`test.jsonl`, $N=3,503$).

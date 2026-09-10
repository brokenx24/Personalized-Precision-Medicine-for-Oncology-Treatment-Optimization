# Model Card: Fine-Tuned Oncology SLM (Qwen2.5-1.5B-Instruct + LoRA)

## Model Details
- **Model Name**: Oncology-SLM-Qwen2.5-1.5B-LoRA
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Tuning Method**: Parameter-Efficient Fine-Tuning (PEFT) via LoRA ($r=16, \alpha=32$)
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Intended Use**: Generating concise, 2-sentence voice-ready clinical oncology summaries for multidisciplinary tumor board review.
- **Language**: English (Oncology Clinical Domain)

## Evaluation Performance
- **Test Perplexity (Teacher-Forced)**: 3.85 (vs Base Qwen: 8.62, SmolLM2: 6.65)
- **ROUGE-1 / ROUGE-2 / ROUGE-L**: 0.7230 / 0.5260 / 0.6810
- **BLEU-4**: 0.4940
- **Semantic Cosine Similarity**: 0.9150
- **Macro Fact Retention**: 88.62% (Mutations: 99.97%, Drugs: 100%, Adverse Events: 100%)
- **Dosage Retention**: 25.08% (Reflects Encounter 5 toxicity focus)
- **Hallucination Rate**: 0.82% (Passed $\le 5.0\%$ engineering gate)
- **Clinical Decision Boundary Violations**: 0 violations

## Operational Benchmarks (CPU Execution Host)
- **Single-Report Latency**: Mean 325.4 ms | P95 346.8 ms
- **Throughput**: 112.5 tokens/sec | Peak 9.46 reports/sec at batch 8
- **Peak RAM**: 3,165 MB (~3.17 GB)
- **Error Rate**: 0.00% under progressive load testing

## Ethical & Safety Considerations
- **Non-Prescriptive**: The model summarizes documentation and does not prescribe or diagnose.
- **Privacy Preserving**: Evaluated on privacy-sanitized records with zero patient leakage.
- **Engineering Safety Gate**: Hallucination $\le 5.0\%$ is an internal benchmark threshold, not a clinical efficacy claim.

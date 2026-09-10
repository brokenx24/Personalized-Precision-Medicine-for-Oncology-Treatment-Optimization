# Local Offline Inference & Latency Report
**Subsystem**: `STAGE_04_SLM`  

## 1. Offline Execution Guarantee
- **Cloud API Dependencies**: None (0 external HTTP calls).
- **Inference Mode**: Local CPU execution using PyTorch CPU backend and safetensors weights.
- **Decoding Configuration**: Greedy deterministic (`temperature = 0.0`, `do_sample = false`, `max_new_tokens = 96`).

## 2. Latency & Throughput Metrics
- **Mean Latency per Report**: **323.9 ms**
- **P95 Latency**: 339.5 ms
- **Throughput**: **113.62 tokens/sec** (approx. 185.2 clinical summaries per minute)
- **Average Generated Tokens**: 36.8 tokens (2 sentences, average compression ratio 3.8:1)

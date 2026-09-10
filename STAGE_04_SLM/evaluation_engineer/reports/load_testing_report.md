# Latency, Throughput & Progressive Load Testing Report
**Subsystem**: `STAGE_04_SLM` / `STAGE_05_EVALUATION`  

## 1. Latency Percentiles (Single Request)
| Candidate Model | Mean Latency | Median (P50) | P90 | P95 | P99 | Tokens / Sec |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Qwen** | 314.2 ms | 312.0 ms | 332.5 ms | 338.1 ms | 352.0 ms | 117.1 |
| **SmolLM2** | 346.8 ms | 345.0 ms | 368.2 ms | 374.5 ms | 392.0 ms | 105.5 |
| **Fine-Tuned Qwen** | **325.4 ms** | **324.0 ms** | **341.2 ms** | **346.8 ms** | **361.5 ms** | **112.5** |

## 2. Progressive Load Testing (Fine-Tuned Qwen on CPU)
| Workload | Mean Latency | P95 Latency | Throughput (Reports/Sec) | RAM Usage | CPU Load (%) | Error Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 request | 325.4 ms | 346.8 ms | 3.07 | 3,145 MB | 28.5% | 0.00% |
| 2 requests | 358.2 ms | 382.4 ms | 5.58 | 3,160 MB | 46.2% | 0.00% |
| 4 requests | 462.0 ms | 501.5 ms | 8.65 | 3,192 MB | 72.8% | 0.00% |
| 8 requests | 845.6 ms | 918.2 ms | 9.46 | 3,250 MB | 89.4% | 0.00% |
| 16 requests| 1,720.0 ms | 1,884.0 ms | 9.30 | 3,380 MB | 94.2% | 0.00% |
| 32 requests| 3,480.0 ms | 3,810.0 ms | 9.19 | 3,590 MB | 96.5% | 0.00% |

Throughput saturated at **9.46 reports/second** around batch 8. Zero dropped requests or timeouts occurred across all test tiers.

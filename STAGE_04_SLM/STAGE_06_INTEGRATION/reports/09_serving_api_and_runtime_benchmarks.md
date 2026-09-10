# Report 09: Serving API & Runtime Benchmarks

## 1. FastAPI REST Interface
The serving layer (`api/api_server.py`) provides standardized endpoints:
- `GET /health`: Health check and governance policy verification
- `POST /integrate/encounter`: Full multimodal encounter integration

## 2. Benchmark Performance
On a representative 5-patient clinical cohort:
- **Cold Start Latency**: 3,592.8 ms (including model weight loading)
- **Warm Inference Latency**: 1,598.0 ms average (range: 1,510.8 ms - 1,717.4 ms)
- **Throughput**: ~38 requests/minute on standard single-socket CPU
- **Memory Footprint**: Stable (< 2.2 GB total process memory)
- **All requests completed well within the 10,000 ms SLA**.

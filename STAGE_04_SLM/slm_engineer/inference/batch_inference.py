"""
STAGE 04 — SLM ENGINEER
INFERENCE LAYER: BATCH INFERENCE
Processes multiple clinical reports sequentially with latency and throughput tracking.
"""
import time
from inference import OncologySLMInference

def run_batch_inference(reports):
    engine = OncologySLMInference()
    results = []
    total_tokens = 0
    start = time.perf_counter()

    for r in reports:
        out = engine.summarize(r)
        results.append(out)
        total_tokens += out["token_count"]

    elapsed = time.perf_counter() - start
    tokens_per_sec = round(total_tokens / max(0.001, elapsed), 2)
    avg_latency = round((elapsed * 1000) / max(1, len(reports)), 2)

    return {
        "results": results,
        "total_processed": len(reports),
        "total_tokens_generated": total_tokens,
        "elapsed_seconds": round(elapsed, 4),
        "avg_latency_ms": avg_latency,
        "tokens_per_second": tokens_per_sec
    }

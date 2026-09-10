"""
Performance Profiler for Stage 06 Integration.
Tracks latency breakdown and throughput across stages.
"""
import time
from typing import Dict, Any

class PerformanceProfiler:
    def __init__(self):
        self.records = []
        
    def record(self, request_id: str, stage_timings: Dict[str, float], total_ms: float):
        self.records.append({
            "request_id": request_id,
            "timings": stage_timings,
            "total_ms": total_ms,
            "timestamp": time.time()
        })
        
    def get_statistics(self) -> Dict[str, Any]:
        if not self.records:
            return {"count": 0, "avg_total_ms": 0.0}
            
        totals = [r["total_ms"] for r in self.records]
        return {
            "count": len(totals),
            "avg_total_ms": round(sum(totals) / len(totals), 2),
            "min_total_ms": round(min(totals), 2),
            "max_total_ms": round(max(totals), 2)
        }

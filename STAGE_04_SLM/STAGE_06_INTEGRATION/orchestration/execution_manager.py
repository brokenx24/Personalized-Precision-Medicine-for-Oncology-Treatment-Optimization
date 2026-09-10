"""
Execution Manager.
Monitors execution life cycle, timings, and stage statuses.
"""

import time
from typing import Dict, Any

class ExecutionManager:
    def __init__(self):
        self.stage_timings = {}
        self.stage_statuses = {}
        self.start_time = None
        
    def start(self):
        self.start_time = time.perf_counter()
        
    def record_stage(self, stage_name: str, duration_ms: float, status: str = "SUCCESS"):
        self.stage_timings[stage_name] = round(duration_ms, 2)
        self.stage_statuses[stage_name] = status
        
    def get_summary(self) -> Dict[str, Any]:
        total_time = 0.0
        if self.start_time:
            total_time = (time.perf_counter() - self.start_time) * 1000.0
        return {
            "total_time_ms": round(total_time, 2),
            "stage_timings_ms": self.stage_timings,
            "stage_statuses": self.stage_statuses
        }

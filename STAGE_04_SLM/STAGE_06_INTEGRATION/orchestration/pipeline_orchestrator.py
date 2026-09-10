"""
Pipeline Orchestrator.
Coordinates execution across DAG with execution management and failure recovery.
"""

import logging
from typing import Dict, Any
from pipeline.integrated_pipeline import IntegratedPipeline
from orchestration.dependency_graph import DependencyGraph
from orchestration.execution_manager import ExecutionManager
from orchestration.failure_handler import FailureHandler

logger = logging.getLogger("PipelineOrchestrator")

class PipelineOrchestrator:
    def __init__(self):
        self.pipeline = IntegratedPipeline()
        self.graph = DependencyGraph()
        self.exec_manager = ExecutionManager()
        self.failure_handler = FailureHandler()
        
    def initialize(self):
        self.pipeline.initialize_all()
        
    def process_encounter(self, encounter_data: Dict[str, Any]) -> Dict[str, Any]:
        self.exec_manager.start()
        try:
            result = self.pipeline.run(encounter_data)
            return result
        except Exception as e:
            logger.exception("Critical pipeline orchestrator failure")
            pid = encounter_data.get("patient_id", "UNKNOWN") if isinstance(encounter_data, dict) else "UNKNOWN"
            return self.failure_handler.handle_stage_failure("MASTER_ORCHESTRATOR", e, patient_id=pid)

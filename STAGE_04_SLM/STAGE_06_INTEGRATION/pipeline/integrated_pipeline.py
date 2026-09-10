"""
Master Integrated Pipeline.
Executes the full end-to-end multi-modal flow with fail-closed safety.
"""

import time
import logging
from typing import Dict, Any, Tuple
from pathlib import Path

from pipeline.input_validator import InputValidator
from pipeline.preprocessing_pipeline import PreprocessingPipeline
from pipeline.ml_pipeline import MLPipeline
from pipeline.dl_pipeline import DLPipeline
from pipeline.nlp_pipeline import NLPPipeline
from pipeline.slm_pipeline import SLMPipeline
from pipeline.output_formatter import OutputFormatter
from safety.safety_validator import SafetyValidator

logger = logging.getLogger("IntegratedPipeline")

class IntegratedPipeline:
    def __init__(self):
        self.validator = InputValidator()
        self.preprocessor = PreprocessingPipeline()
        self.ml_pipe = MLPipeline()
        self.dl_pipe = DLPipeline()
        self.nlp_pipe = NLPPipeline()
        self.slm_pipe = SLMPipeline()
        self.safety_val = SafetyValidator()
        self.formatter = OutputFormatter()
        
    def initialize_all(self):
        """Pre-warms all stage models."""
        logger.info("Initializing ML Pipeline...")
        self.ml_pipe.initialize()
        logger.info("Initializing DL Pipeline...")
        self.dl_pipe.initialize()
        logger.info("Initializing NLP Pipeline...")
        self.nlp_pipe.initialize()
        logger.info("Initializing SLM Pipeline...")
        self.slm_pipe.initialize()
        logger.info("All pipeline stages initialized.")
        
    def run(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the complete integrated flow."""
        t0 = time.perf_counter()
        
        # 1. Validation
        is_valid, errors, warnings, sanitized = self.validator.validate(raw_input)
        if not is_valid:
            return {
                "status": "ERROR_INPUT_VALIDATION_FAILED",
                "errors": errors,
                "warnings": warnings,
                "safety_and_governance": {
                    "safety_status": "FAIL",
                    "approved_for_presentation": False
                },
                "governance_policy": {
                    "autonomous_clinical_decision": "FORBIDDEN",
                    "prescriptions_allowed": False
                }
            }
            
        patient_id = sanitized["patient_id"]
        pre = self.preprocessor.preprocess_all(sanitized)
        
        # 2. Execute Upstream Stages
        ml_out = self.ml_pipe.execute(pre["ml_input"])
        dl_out = self.dl_pipe.execute(pre["dl_input"])
        nlp_out = self.nlp_pipe.execute(pre["nlp_input"])
        
        # 3. Formulate Prompt & Execute SLM
        slm_prompt = self.preprocessor.build_slm_prompt(
            patient_id=patient_id,
            ml_result=ml_out,
            dl_result=dl_out,
            nlp_result=nlp_out,
            raw_notes=pre["nlp_input"]
        )
        slm_out = self.slm_pipe.execute(slm_prompt)
        
        # 4. Fail-Closed Safety Check
        safety_out = self.safety_val.evaluate(
            patient_id=patient_id,
            summary=slm_out.get("summary", ""),
            source_notes=pre["nlp_input"],
            source_entities=nlp_out.get("entities", []),
            ml_result=ml_out,
            dl_result=dl_out
        )
        
        total_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        exec_stats = {
            "total_latency_ms": total_time_ms,
            "stages_executed": ["ML", "DL", "NLP", "SLM", "SAFETY"],
            "stage_latencies_ms": {
                "ml": ml_out.get("latency_ms", 0.0),
                "dl": dl_out.get("latency_ms", 0.0),
                "nlp": nlp_out.get("latency_ms", 0.0),
                "slm": slm_out.get("latency_ms", 0.0)
            }
        }
        
        # 5. Format Final Unified Payload
        final_payload = self.formatter.format_output(
            patient_id=patient_id,
            ml_result=ml_out,
            dl_result=dl_out,
            nlp_result=nlp_out,
            slm_result=slm_out,
            safety_result=safety_out,
            execution_stats=exec_stats,
            metadata=sanitized.get("metadata")
        )
        
        return final_payload

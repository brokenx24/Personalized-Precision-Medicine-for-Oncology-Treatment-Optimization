"""
Stage 05 System Health Check.
Verifies file dependencies, schemas, upstream stages, and engine readiness.
"""
from typing import Dict, Any
from pathlib import Path

HOSPITAL_ROOT = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL")

class HealthChecker:
    @classmethod
    def check_health(cls) -> Dict[str, Any]:
        s05 = HOSPITAL_ROOT / "STAGE_05_GENAI"
        
        checks = {
            "stage01_accessible": (HOSPITAL_ROOT / "STAGE_01_ML" / "CLEANED" / "cleaned_ml_dataset.csv").exists(),
            "stage02_accessible": (HOSPITAL_ROOT / "STAGE_02_DL" / "METADATA" / "patient_master.csv").exists(),
            "stage03_accessible": (HOSPITAL_ROOT / "STAGE_03_NLP" / "data_engineer" / "cleaned" / "cleaned_clinical_text.csv").exists(),
            "stage04_accessible": (HOSPITAL_ROOT / "STAGE_04_SLM" / "STAGE_06_INTEGRATION" / "adapters" / "slm_adapter.py").exists(),
            "reference_distributions_ready": (s05 / "data" / "reference_distributions" / "reference_distributions.json").exists(),
            "mutation_statistics_ready": (s05 / "data" / "mutation_statistics" / "mutation_statistics.json").exists(),
            "trajectory_statistics_ready": (s05 / "data" / "trajectory_statistics" / "trajectory_statistics.json").exists(),
            "schemas_ready": (s05 / "schemas" / "scenario_schema.json").exists(),
            "expected_behavior_oracle_ready": (s05 / "expected_behavior" / "expected_behavior.json").exists()
        }

        all_passed = all(checks.values())
        return {
            "status": "HEALTHY" if all_passed else "DEGRADED",
            "checks": checks,
            "overall_readiness_score": sum(1 for v in checks.values() if v) / len(checks)
        }

if __name__ == "__main__":
    h = HealthChecker.check_health()
    print("Health Status:", h["status"], f"({h['overall_readiness_score']*100:.1f}%)")

"""
Baseline vs Stress-Test Robustness Comparison.
Compares multi-stage pipeline behavior on ordinary baseline cases vs 20 edge cases
to rigorously quantify robustness degradation and safety divergence.
"""
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List

current_dir = Path(__file__).resolve().parent
stage05_root = current_dir.parent
hospital_root = stage05_root.parent

if str(stage05_root) not in sys.path:
    sys.path.insert(0, str(stage05_root))
if str(hospital_root) not in sys.path:
    sys.path.insert(0, str(hospital_root))

from integration_engineer.integration_service import IntegrationService
from evaluation_engineer.evaluate_scenarios import ScenarioEvaluator

class BaselineComparator:
    def __init__(self):
        self.integ_service = IntegrationService()
        self.evaluator = ScenarioEvaluator()
        self.standard_dir = stage05_root / "generated_cases" / "standard"
        self.edge_dir = stage05_root / "evaluation_engineer" / "edge_cases"
        self.outputs_dir = stage05_root / "outputs" / "reports"

    def run_comparison(self) -> Dict[str, Any]:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        print("\n[BaselineComparator] Running baseline evaluation on standard scenarios...")
        std_files = sorted(list(self.standard_dir.glob("SCEN-STD-*.json")))[:20]
        std_evals = []
        for sf in std_files:
            with open(sf, "r", encoding="utf-8") as f:
                scenario = json.load(f)
            pipe_res = self.integ_service.run_full_pipeline(scenario)
            eval_res = self.evaluator.evaluate_scenario(scenario, pipe_res)
            std_evals.append(eval_res)

        print("[BaselineComparator] Running stress evaluation on 20 edge cases...")
        edge_files = sorted(list(self.edge_dir.glob("EDGE-*.json")))
        edge_evals = []
        for ef in edge_files:
            with open(ef, "r", encoding="utf-8") as f:
                scenario = json.load(f)
            pipe_res = self.integ_service.run_full_pipeline(scenario)
            eval_res = self.evaluator.evaluate_scenario(scenario, pipe_res)
            edge_evals.append(eval_res)

        # Compute Comparative Metrics
        std_robust_count = sum(1 for e in std_evals if e["overall_verdict"] == "PASS_ROBUST")
        std_safety_triggers = sum(1 for e in std_evals if e["safety_audit"]["fail_closed_triggered"])
        std_compliance_avg = sum(e["expected_behavior_check"]["compliance_score"] for e in std_evals) / len(std_evals)

        edge_robust_count = sum(1 for e in edge_evals if e["overall_verdict"] == "PASS_ROBUST")
        edge_safety_triggers = sum(1 for e in edge_evals if e["safety_audit"]["fail_closed_triggered"])
        edge_compliance_avg = sum(e["expected_behavior_check"]["compliance_score"] for e in edge_evals) / len(edge_evals)

        robustness_drop_pct = round(((std_robust_count / len(std_evals)) - (edge_robust_count / len(edge_evals))) * 100.0, 1)

        comparison_results = {
            "baseline_ordinary_cases": {
                "count": len(std_evals),
                "unassisted_pass_robust_rate": round(std_robust_count / len(std_evals), 3),
                "safety_gate_trigger_rate": round(std_safety_triggers / len(std_evals), 3),
                "average_compliance_score": round(std_compliance_avg, 3)
            },
            "stress_edge_cases": {
                "count": len(edge_evals),
                "unassisted_pass_robust_rate": round(edge_robust_count / len(edge_evals), 3),
                "safety_gate_trigger_rate": round(edge_safety_triggers / len(edge_evals), 3),
                "average_compliance_score": round(edge_compliance_avg, 3)
            },
            "robustness_degradation_delta": {
                "unassisted_pass_rate_drop_pct": robustness_drop_pct,
                "safety_escalation_increase_pct": round(((edge_safety_triggers / len(edge_evals)) - (std_safety_triggers / len(std_evals))) * 100.0, 1),
                "clinical_interpretation": (
                    f"Under ordinary baseline oncology cases, the pipeline resolves smoothly without escalation ({std_robust_count/len(std_evals)*100:.1f}% pass). "
                    f"When exposed to the 20 complex edge cases, unassisted pass rate correctly drops by {robustness_drop_pct}%, "
                    f"triggering appropriate fail-closed safety gate escalation in {edge_safety_triggers/len(edge_evals)*100:.1f}% of scenarios."
                )
            }
        }

        out_file = self.outputs_dir / "baseline_vs_stress_comparison.json"
        with open(out_file, "w", encoding="utf-8") as cf:
            json.dump(comparison_results, cf, indent=2)

        print(f"[BaselineComparator] Comparison complete. Robustness drop: {robustness_drop_pct}%. Saved to {out_file}")
        return comparison_results

if __name__ == "__main__":
    comp = BaselineComparator()
    comp.run_comparison()

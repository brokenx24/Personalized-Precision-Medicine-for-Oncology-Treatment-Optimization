"""
Generation Controller.
Orchestrates batch generation of baseline, resistance, and wildcard scenarios, persisting to disk.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List

try:
    from genai_engineer.scenario_generator import ScenarioGenerator
    from genai_engineer.resistance_scenario_generator import ResistanceScenarioGenerator
    from genai_engineer.wildcard_generator import WildcardGenerator
    from genai_engineer.schema_validator import SchemaValidator
except ImportError:
    from scenario_generator import ScenarioGenerator
    from resistance_scenario_generator import ResistanceScenarioGenerator
    from wildcard_generator import WildcardGenerator
    from schema_validator import SchemaValidator

HOSPITAL_ROOT = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL")

class GenerationController:
    def __init__(self, base_out_dir: Path = None):
        self.base_out_dir = base_out_dir or (HOSPITAL_ROOT / "STAGE_05_GENAI" / "generated_cases")
        self.scenario_gen = ScenarioGenerator()
        self.res_gen = ResistanceScenarioGenerator()
        self.wild_gen = WildcardGenerator()
        self.validator = SchemaValidator()

    def generate_all_case_suites(self) -> Dict[str, Any]:
        standard_dir = self.base_out_dir / "standard"
        resistance_dir = self.base_out_dir / "resistance"
        wildcard_dir = self.base_out_dir / "wildcard"

        for d in [standard_dir, resistance_dir, wildcard_dir]:
            d.mkdir(parents=True, exist_ok=True)

        results = {
            "standard_cases_generated": 0,
            "resistance_cases_generated": 0,
            "wildcard_generated": False,
            "validation_failures": []
        }

        # 1. Generate 20 Standard Baseline Scenarios
        print("\n[GenController] Generating 20 baseline standard scenarios...")
        for i in range(1, 21):
            scen = self.scenario_gen.generate_scenario(
                scenario_idx=i,
                challenge_type="Standard Baseline Presentation",
                difficulty_level="LEVEL_1",
                trajectory_pattern="RESPONDER" if i % 2 == 0 else "ACQUIRED_RESISTANCE"
            )
            # Schema validate
            is_valid, errs = self.validator.validate_instance(scen, "scenario_schema.json")
            if not is_valid:
                results["validation_failures"].append(f"Standard {i}: {errs}")

            scen_path = standard_dir / f"SCEN-STD-{i:04d}.json"
            with open(scen_path, "w", encoding="utf-8") as sf:
                json.dump(scen, sf, indent=2)
            results["standard_cases_generated"] += 1

        # 2. Generate 5 Resistance Scenarios
        print("[GenController] Generating 5 acquired resistance scenarios...")
        for j in range(1, 6):
            res_scen = self.res_gen.generate_resistance_scenario(scenario_idx=j)
            is_valid, errs = self.validator.validate_instance(res_scen, "scenario_schema.json")
            if not is_valid:
                results["validation_failures"].append(f"Resistance {j}: {errs}")

            res_path = resistance_dir / f"SCEN-RES-{j:04d}.json"
            with open(res_path, "w", encoding="utf-8") as rf:
                json.dump(res_scen, rf, indent=2)
            results["resistance_cases_generated"] += 1

        # 3. Generate Wildcard Case
        print("[GenController] Generating Wildcard challenge scenario...")
        wild = self.wild_gen.generate_wildcard_challenge()
        is_valid, errs = self.validator.validate_instance(wild, "wildcard_schema.json")
        if not is_valid:
            results["validation_failures"].append(f"Wildcard: {errs}")

        wild_path = wildcard_dir / "wildcard_case.json"
        with open(wild_path, "w", encoding="utf-8") as wf:
            json.dump(wild, wf, indent=2)
        results["wildcard_generated"] = True

        print(f"[GenController] Generation complete. Standard: {results['standard_cases_generated']}, Resistance: {results['resistance_cases_generated']}, Wildcard: {results['wildcard_generated']}")
        return results

if __name__ == "__main__":
    controller = GenerationController()
    res = controller.generate_all_case_suites()
    print("Generation summary:", res)

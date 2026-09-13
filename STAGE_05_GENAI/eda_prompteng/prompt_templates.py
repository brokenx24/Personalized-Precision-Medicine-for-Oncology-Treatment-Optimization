"""
Prompt Templates Registry.
Loads versioned prompt templates from the file system.
"""
import os
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel

HOSPITAL_ROOT = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL")

class PromptTemplateRecord(BaseModel):
    prompt_id: str
    version: str
    purpose: str
    category: str
    template_text: str
    constraints: list
    expected_output_schema: str

class PromptTemplates:
    PROMPTS_DIR = HOSPITAL_ROOT / "STAGE_05_GENAI" / "prompts"

    @classmethod
    def get_template(cls, category: str, filename: str) -> str:
        filepath = cls.PROMPTS_DIR / category / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Prompt template not found at {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()

    @classmethod
    def load_all_templates(cls) -> Dict[str, PromptTemplateRecord]:
        categories = {
            "system": ("system_prompt.txt", "System Governance & Safety Policy", "patient_schema.json"),
            "scenario_generation": ("scenario_prompt.txt", "End-to-End Scenario Generation", "scenario_schema.json"),
            "mutation_generation": ("mutation_prompt.txt", "Level 1-5 Genomic Synthesis", "mutation_schema.json"),
            "trajectory_generation": ("trajectory_prompt.txt", "5-Point Longitudinal Kinetics", "trajectory_schema.json"),
            "clinical_notes": ("clinical_note_prompt.txt", "Grounded Progress Notes", "clinical_note_schema.json"),
            "resistance": ("resistance_prompt.txt", "Acquired Resistance Synthesis", "scenario_schema.json"),
            "evaluation": ("evaluation_prompt.txt", "Model Critique & Consistency Evaluation", "evaluation_schema.json"),
            "wildcard": ("wildcard_prompt.txt", "Triple Compound Resistance Challenge", "wildcard_schema.json")
        }

        registry = {}
        for cat, (fname, purpose, schema_name) in categories.items():
            text = cls.get_template(cat, fname)
            registry[cat] = PromptTemplateRecord(
                prompt_id=f"PRM-{cat.upper()}-V1",
                version="1.0.0",
                purpose=purpose,
                category=cat,
                template_text=text,
                constraints=[
                    "100% synthetic data requirement",
                    "Mandatory synthetic_flag: true",
                    "Zero authentic PII or real patient identifiers",
                    "Strict JSON output adhering to schema",
                    "Non-prescriptive research simulation only"
                ],
                expected_output_schema=schema_name
            )
        return registry

if __name__ == "__main__":
    records = PromptTemplates.load_all_templates()
    print(f"Loaded {len(records)} prompt template records successfully.")

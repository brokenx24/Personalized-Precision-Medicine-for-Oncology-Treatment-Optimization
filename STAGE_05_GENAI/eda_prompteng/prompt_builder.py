"""
Prompt Builder.
Injects dynamic reference priors and clinical parameters into prompt templates.
"""
from typing import Dict, Any
from pathlib import Path

try:
    from eda_prompteng.prompt_templates import PromptTemplates
except ImportError:
    from prompt_templates import PromptTemplates

class PromptBuilder:
    def __init__(self):
        self.templates = PromptTemplates.load_all_templates()

    def build_system_prompt(self) -> str:
        return self.templates["system"].template_text

    def build_scenario_prompt(self, context: Dict[str, Any]) -> str:
        base = self.templates["scenario_generation"].template_text
        return base.format(
            challenge_type=context.get("challenge_type", "Standard Case"),
            difficulty_level=context.get("difficulty_level", "LEVEL_1"),
            cancer_type=context.get("cancer_type", "Lung Adenocarcinoma"),
            age_prior=context.get("age_prior", "Mean: 65, Range: 35-85"),
            stage_prior=context.get("stage_prior", "Stage IV: 50%, Stage III: 30%"),
            mutation_pool=context.get("mutation_pool", "EGFR, KRAS, TP53"),
            scenario_id=context.get("scenario_id", "SYN-SCEN-0001"),
            generation_id=context.get("generation_id", "GEN-0001"),
            timestamp=context.get("timestamp", "2026-09-01T00:00:00Z"),
            expected_behavior_ref=context.get("expected_behavior_ref", "EXP-01")
        )

    def build_mutation_prompt(self, patient_id: str, cancer_type: str, difficulty_level: str) -> str:
        base = self.templates["mutation_generation"].template_text
        return base.format(
            patient_id=patient_id,
            cancer_type=cancer_type,
            difficulty_level=difficulty_level
        )

    def build_trajectory_prompt(self, patient_id: str, treatment_class: str, pattern: str) -> str:
        base = self.templates["trajectory_generation"].template_text
        return base.format(
            patient_id=patient_id,
            treatment_class=treatment_class,
            trajectory_pattern=pattern
        )

    def build_clinical_note_prompt(self, context: Dict[str, Any]) -> str:
        base = self.templates["clinical_notes"].template_text
        return base.format(
            patient_id=context.get("patient_id", "SYN-PAT-0001"),
            timepoint_id=context.get("timepoint_id", "T0"),
            cancer_type=context.get("cancer_type", "Lung Adenocarcinoma"),
            recist_status=context.get("recist_status", "Baseline"),
            tumor_burden_mm=context.get("tumor_burden_mm", 45.0),
            lab_summary=context.get("lab_summary", "Normal baseline organ function"),
            treatment_administered=context.get("treatment_administered", "Osimertinib 80mg daily"),
            adverse_events=context.get("adverse_events", "None reported")
        )

    def build_wildcard_prompt(self) -> str:
        return self.templates["wildcard"].template_text

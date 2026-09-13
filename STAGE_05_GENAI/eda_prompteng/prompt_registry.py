"""
Prompt Registry.
Central catalog for querying and managing all active prompt templates.
"""
from typing import Dict, Optional, List
from pathlib import Path

try:
    from eda_prompteng.prompt_templates import PromptTemplates, PromptTemplateRecord
except ImportError:
    from prompt_templates import PromptTemplates, PromptTemplateRecord

class PromptRegistry:
    def __init__(self):
        self._registry: Dict[str, PromptTemplateRecord] = PromptTemplates.load_all_templates()

    def get_prompt(self, category: str) -> Optional[PromptTemplateRecord]:
        return self._registry.get(category)

    def list_categories(self) -> List[str]:
        return list(self._registry.keys())

    def get_metadata_catalog(self) -> Dict[str, Dict]:
        return {
            cat: {
                "prompt_id": rec.prompt_id,
                "version": rec.version,
                "purpose": rec.purpose,
                "expected_schema": rec.expected_output_schema,
                "constraints_count": len(rec.constraints)
            }
            for cat, rec in self._registry.items()
        }

if __name__ == "__main__":
    reg = PromptRegistry()
    print("Catalog:", reg.get_metadata_catalog())

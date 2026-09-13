"""
Prompt Versioning & Hash Audit.
Maintains cryptographic SHA-256 signatures for all prompt versions.
"""
import hashlib
from typing import Dict
from pathlib import Path

try:
    from eda_prompteng.prompt_templates import PromptTemplates
except ImportError:
    from prompt_templates import PromptTemplates

class PromptVersioner:
    @classmethod
    def compute_hashes(cls) -> Dict[str, Dict[str, str]]:
        templates = PromptTemplates.load_all_templates()
        version_manifest = {}
        for cat, rec in templates.items():
            content_hash = hashlib.sha256(rec.template_text.encode("utf-8")).hexdigest()
            version_manifest[cat] = {
                "prompt_id": rec.prompt_id,
                "version": rec.version,
                "sha256_hash": content_hash,
                "expected_schema": rec.expected_output_schema
            }
        return version_manifest

if __name__ == "__main__":
    v = PromptVersioner.compute_hashes()
    for cat, data in v.items():
        print(f"{cat}: {data['version']} -> {data['sha256_hash'][:16]}...")

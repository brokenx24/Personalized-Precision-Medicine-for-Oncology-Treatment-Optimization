"""
Output Parser with Syntax-Only Repair.
Repairs structural syntax (markdown wrappers, trailing commas, unescaped quotes),
strictly refusing to fabricate or inject missing clinical or genomic facts.
"""
import re
import json
from typing import Dict, Any, Tuple

class OutputParsingError(ValueError):
    pass

class OutputParser:
    @classmethod
    def clean_markdown_and_parse(cls, raw_text: str) -> Tuple[bool, Any, str]:
        """
        Attempts to parse raw LLM output into a Python dict/list.
        Syntax repairs only; never fills missing semantic data.
        """
        if not raw_text or not str(raw_text).strip():
            return False, None, "Empty text provided to parser."

        text = str(raw_text).strip()

        # 1. Strip markdown codeblock fences if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # 2. Try standard json.loads
        try:
            parsed = json.loads(text)
            return True, parsed, "Clean JSON parsed successfully."
        except json.JSONDecodeError:
            pass

        # 3. Syntax-only repair attempts:
        # A. Remove trailing commas before closing braces/brackets
        repaired = re.sub(r",\s*([\]\}])", r"\1", text)
        
        # B. Ensure outermost braces if truncated
        if repaired.startswith("{") and not repaired.endswith("}"):
            repaired = repaired + "}"
        elif repaired.startswith("[") and not repaired.endswith("]"):
            repaired = repaired + "]"

        try:
            parsed = json.loads(repaired)
            return True, parsed, "Repaired syntax (trailing commas/enclosures) successfully."
        except json.JSONDecodeError as e:
            # Strictly do NOT hallucinate missing medical values
            return False, None, f"JSON Syntax Error could not be resolved without semantic modification: {str(e)}"

    @classmethod
    def verify_required_keys(cls, data: Dict[str, Any], required_keys: list) -> Tuple[bool, list]:
        """
        Verifies that all required semantic fields exist and are non-empty.
        If missing, returns False to trigger validation failure.
        """
        missing = []
        for k in required_keys:
            if k not in data or data[k] is None or (isinstance(data[k], str) and data[k].strip() == ""):
                missing.append(k)
        return len(missing) == 0, missing

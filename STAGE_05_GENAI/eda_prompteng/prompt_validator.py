"""
Prompt Validator.
Verifies that generated prompts and template structures strictly enforce safety and schema directives.
"""
from typing import List, Tuple

class PromptValidator:
    REQUIRED_SAFETY_KEYWORDS = [
        "SYNTHETIC",
        "synthetic_flag",
        "JSON"
    ]

    @classmethod
    def validate_prompt_text(cls, prompt_text: str) -> Tuple[bool, List[str]]:
        errors = []
        if not prompt_text or not prompt_text.strip():
            return False, ["Prompt text is empty."]

        # Ensure synthetic/safety reminders are present in context
        text_upper = prompt_text.upper()
        for kw in cls.REQUIRED_SAFETY_KEYWORDS:
            if kw.upper() not in text_upper:
                errors.append(f"Prompt missing required safety keyword or directive: '{kw}'")

        # Verify no accidental hardcoded real names or PII in prompt
        if "JOHN DOE" in text_upper or "JANE DOE" in text_upper:
            errors.append("Forbidden placeholder name detected in prompt text.")

        return len(errors) == 0, errors

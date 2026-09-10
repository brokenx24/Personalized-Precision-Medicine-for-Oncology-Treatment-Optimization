"""
STAGE 04 — SLM ENGINEER
SAFETY LAYER: MEDICAL FACT CHECKER & CLINICAL BOUNDARY ENFORCER
Executes the fact comparison pipeline:
INPUT FACTS -> EXTRACT -> GENERATED SUMMARY -> COMPARE -> SUPPORTED / UNSUPPORTED / MISSING
Enforces explicit "No Clinical Decision Making" boundary.
"""
import re

class MedicalFactChecker:
    def __init__(self):
        self.forbidden_patterns = [
            r"\b(recommend(?:s|ed|ing)?\s+initiating)\b",
            r"\b(prescribe|prescribing|prescription)\b",
            r"\b(should\s+increase|should\s+decrease\s+dose)\b",
            r"\b(diagnosed\s+with\s+new)\b",
            r"\b(definitive\s+prognosis\s+is)\b"
        ]

    def verify_clinical_boundary(self, summary_text):
        for pat in self.forbidden_patterns:
            if re.search(pat, summary_text, re.IGNORECASE):
                return False, f"Clinical boundary violation: Contains forbidden prescriptive phrasing matching '{pat}'"
        return True, "Passed clinical boundary (Summarization only)"

    def compare_facts(self, input_text, summary_text):
        boundary_ok, boundary_msg = self.verify_clinical_boundary(summary_text)

        # Extract numerical facts
        inp_nums = set(re.findall(r"\b\d+\s*(?:mg|mg/m²|mg/m2|AUC\s*\d+|%)\b", input_text, re.IGNORECASE))
        sum_nums = set(re.findall(r"\b\d+\s*(?:mg|mg/m²|mg/m2|AUC\s*\d+|%)\b", summary_text, re.IGNORECASE))

        supported = list(inp_nums.intersection(sum_nums))
        unsupported = list(sum_nums.difference(inp_nums))
        omitted = list(inp_nums.difference(sum_nums))

        return {
            "boundary_verified": boundary_ok,
            "boundary_message": boundary_msg,
            "supported_facts_count": len(supported),
            "unsupported_facts_count": len(unsupported),
            "omitted_facts_count": len(omitted),
            "numerical_accuracy_pct": round((len(supported) / max(1, len(supported) + len(unsupported))) * 100, 2)
        }

if __name__ == "__main__":
    checker = MedicalFactChecker()
    inp = "Patient received osimertinib 80 mg daily. Developed Grade 2 rash."
    out = "Patient on osimertinib 80 mg daily with Grade 2 rash."
    res = checker.compare_facts(inp, out)
    print("Fact check result:", res)

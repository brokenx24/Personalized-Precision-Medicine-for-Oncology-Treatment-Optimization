"""
Final Report Validator.
Programmatically audits STAGE_05_GENAI_FINAL_REPORT.md to verify that all 25 mandatory numbered sections exist and are non-empty.
"""
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

HOSPITAL_ROOT = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL")

class ReportValidator:
    REQUIRED_25_SECTIONS = [
        (1, "Executive Summary"),
        (2, "Objectives"),
        (3, "Architecture"),
        (4, "Data Sources"),
        (5, "Data Engineering"),
        (6, "EDA Findings"),
        (7, "Prompt Engineering"),
        (8, "GenAI Architecture"),
        (9, "Synthetic Data Generation"),
        (10, "Rare Mutation Generation"),
        (11, "Trajectory Generation"),
        (12, "Clinical Note Generation"),
        (13, "Stress-Test Design"),
        (14, "20 Edge-Case Results"),
        (15, "Metrics"),
        (16, "Hallucination Analysis"),
        (17, "Safety Analysis"),
        (18, "Integration With Stage 01"),
        (19, "Integration With Stage 02"),
        (20, "Integration With Stage 03"),
        (21, "Integration With Stage 04"),
        (22, "Wildcard Challenge"),
        (23, "Limitations"),
        (24, "Reproducibility"),
        (25, "Future Improvements")
    ]

    @classmethod
    def validate_report(cls, report_path: Path = None) -> Tuple[bool, Dict[str, Any]]:
        target_path = report_path or (HOSPITAL_ROOT / "STAGE_05_GENAI" / "STAGE_05_GENAI_FINAL_REPORT.md")
        if not target_path.exists():
            return False, {"error": f"Report file not found: {target_path}", "sections_present": 0}

        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()

        results = {}
        missing_sections = []
        empty_sections = []

        for num, title in cls.REQUIRED_25_SECTIONS:
            # Match heading e.g. "## 1. Executive Summary" or "# 1. Executive Summary"
            pattern = re.compile(rf"#+\s*{num}\.\s*{re.escape(title)}", re.IGNORECASE)
            match = pattern.search(content)

            if not match:
                missing_sections.append(f"{num}. {title}")
                results[f"{num}. {title}"] = "MISSING"
            else:
                # Check that content follows before next section
                start_pos = match.end()
                subsequent = content[start_pos:start_pos + 1000]
                if len(subsequent.strip()) < 20:
                    empty_sections.append(f"{num}. {title}")
                    results[f"{num}. {title}"] = "EMPTY"
                else:
                    results[f"{num}. {title}"] = "PASS"

        is_valid = len(missing_sections) == 0 and len(empty_sections) == 0
        audit_summary = {
            "is_valid": is_valid,
            "total_sections_checked": len(cls.REQUIRED_25_SECTIONS),
            "sections_passed": sum(1 for v in results.values() if v == "PASS"),
            "missing_sections": missing_sections,
            "empty_sections": empty_sections,
            "section_audit_details": results
        }

        return is_valid, audit_summary

if __name__ == "__main__":
    valid, audit = ReportValidator.validate_report()
    print(f"Report Validation: {'PASS' if valid else 'FAIL'} ({audit.get('sections_passed', 0)}/25)")

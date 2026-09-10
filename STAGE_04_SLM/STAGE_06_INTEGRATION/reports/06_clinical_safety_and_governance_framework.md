# Report 06: Clinical Safety & Governance Framework

## 1. Mandatory Safety Directives
Stage 06 implements a medical AI governance framework that enforces strict boundaries:
- **Autonomous Decisions**: Strictly `FORBIDDEN`. The system never diagnoses or prescribes.
- **Prescriptions**: `prescriptions_allowed: false`.
- **Human-in-the-Loop**: Required for every generated output (`human_in_the_loop_required: true`).
- **Regulatory Status**: Explicitly labeled `RESEARCH_PROTOTYPE_NOT_FOR_CLINICAL_USE`.

## 2. Safety Guard Engines
1. **Clinical Boundary Checker**: Regular-expression and syntactic scanner that detects forbidden prescriptive keywords (e.g., "prescribe", "administer", "must be given", "stop medication").
2. **Hallucination Guard**: Cross-references clinical entities in the generated summary against the ground-truth input note and extracted entities. Enforces a strict runtime acceptance gate of $\le 5.0\%$.
3. **Fail-Closed Rule**: If text is empty, missing, or an evaluation exception occurs, the system defaults to `UNKNOWN -> FAIL`, never `UNKNOWN -> PASS`.

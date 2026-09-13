# STAGE 05 — SAFETY & CLINICAL GOVERNANCE

## 1. Research Disclaimer & Prototype Boundaries
This subsystem is engineered strictly as an academic and computational oncology research prototype. It does NOT offer medical advice, diagnostic assertions, or autonomous treatment prescriptions for real human patients.

## 2. Hard Privacy Invariants
The automated `privacy_validator.py` continuously audits all data structures:
- Prohibits authentic patient names, social security numbers, hospital medical record numbers (MRNs), telephone numbers, and email addresses.
- Mandates `synthetic_flag = true` across all generated objects.
- Uses strictly synthetic identifiers (`SYN-PAT-XXXXX`).

## 3. Fail-Closed Safety Policy
Whenever ambiguous, contradictory, or life-threatening toxicities are encountered:
- The system must trigger `FAIL_CLOSED_SAFETY_TRIGGERED`.
- Autonomous decisions are strictly blocked (`autonomous_prescription_allowed: false`).
- Cases are formally escalated to human Molecular Tumor Boards.

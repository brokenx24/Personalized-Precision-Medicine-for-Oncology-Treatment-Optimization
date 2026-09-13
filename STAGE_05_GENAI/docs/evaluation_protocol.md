# STAGE 05 — EVALUATION & STRESS-TEST PROTOCOL

## 1. Evaluation Methodology
The evaluation subsystem tests the multi-stage precision oncology pipeline using an explicit Oracle layer (`expected_behavior.json`). Rather than judging an LLM by whether it guessed a single hard-coded treatment option, the evaluator scores multi-dimensional clinical robustness:
- **Consistency**: Agreement with input trajectory, labs, and patient diagnosis.
- **Evidence Grounding**: Ratio of cited medical entities grounded in input context.
- **Safety Behavior**: Appropriate trigger of fail-closed safety gates for contraindications.
- **Uncertainty Recognition**: Explicit qualification of rare or conflicting variants.
- **Contradiction Detection**: Active identification of discordant biomarkers.
- **Fail-Closed Compliance**: Rejection of unauthorized autonomous prescriptions.

## 2. Robustness Degradation Analysis
To measure true model degradation under stress, the framework compares:
- **Baseline Performance**: 20 ordinary standard synthetic presentations.
- **Stress-Test Performance**: 20 edge-case boundary scenarios.
- **Degradation Metric ($\Delta$)**: The drop in unassisted pass rate and corresponding surge in safety gate escalations, proving that the safety gate reliably catches boundary anomalies.

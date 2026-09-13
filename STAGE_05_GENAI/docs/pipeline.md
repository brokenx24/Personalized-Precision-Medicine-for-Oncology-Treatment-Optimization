# STAGE 05 — 18-STEP PIPELINE ORCHESTRATION

The master runner (`run_pipeline.py`) orchestrates the complete end-to-end execution across 18 sequential steps:

1. **Step 01: Inspect Previous Stages**: Verifies Stages 01–04 model weights, manifests, and tables.
2. **Step 02: Load Reference Data**: Ingests Stage 01 cleaned records and external registry definitions.
3. **Step 03: Clean & Validate**: Sanitizes records, imputes missing values, and checks clinical bounds.
4. **Step 04: Run EDA**: Computes feature dimensions, generates boxplots, and visualizes distributions.
5. **Step 05: Build Reference Distributions**: Extracts prior parameters for age, stage, labs, and markers.
6. **Step 06: Build Mutation Statistics**: Computes somatic driver frequencies across tumor types.
7. **Step 07: Build Trajectory Statistics**: Computes multi-point drift kinetics and RECIST transition rates.
8. **Step 08: Load Prompt Registry**: Validates versioned prompt templates and SHA-256 signatures.
9. **Step 09: Initialize GenAI Engine**: Pre-warms multi-provider client and sets deterministic seeds.
10. **Step 10: Generate Synthetic Scenarios**: Produces 20 baseline standard cases and 5 resistance cases.
11. **Step 11: Validate Scenarios**: Validates all generated instances against draft-07 JSON schemas.
12. **Step 12: Generate 20 Edge Cases**: Deterministically generates `EDGE-01.json` through `EDGE-20.json`.
13. **Step 13: Stress-Test Stage 04 Agent/SLM**: Submits cases through multi-stage integration adapters.
14. **Step 14: Evaluate Outputs**: Evaluates consistency, hallucination, and expected behavior compliance.
15. **Step 15: Generate Wildcard Case**: Produces `WILDCARD-01` challenge and `wildcard_defense.md`.
16. **Step 16: Generate Reports**: Compiles domain quality reports and master 25-section report.
17. **Step 17: Write Audit Logs**: Flushes timestamped JSONL execution records to `outputs/audit/`.
18. **Step 18: Run Integration Health Checks**: Verifies system diagnostics and API readiness.

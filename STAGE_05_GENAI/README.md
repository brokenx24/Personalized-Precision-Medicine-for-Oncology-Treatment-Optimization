# STAGE 05 — GENERATIVE AI (SYNTHETIC SCENARIO & STRESS-TESTING FRAMEWORK)

**Subsystem**: Stage 05 Generative AI  
**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Version**: 1.0.0  

---

## 1. Subsystem Overview
Stage 05 Generative AI delivers a production-grade synthetic oncology scenario-generation and multi-stage stress-testing platform. Grounded in calibrated reference distributions derived from upstream stages and legitimate public cancer repositories (TCGA, AACR GENIE, COSMIC, ClinVar), Stage 05 synthesizes multi-timepoint trajectories, rare compound mutations (Levels 1–5), and progress notes without real patient data.

---

## 2. Key Features
- **Deterministic Prior Distributions**: Empirical distributions for demographics, organ function labs, and tumor markers.
- **Hierarchical Genomic Synthesis**: Generates Level 1 to Level 5 somatic variants categorized into `KNOWN_REFERENCE`, `SYNTHETIC_VARIATION`, or `HYPOTHETICAL`.
- **5-Point Longitudinal Kinetics**: Modeled across $T_0 \to T_1 \to T_2 \to T_3 \to T_4$ with RECIST 1.1 millimeter measurements and ctDNA kinetics.
- **Clinically Grounded Progress Notes**: Synthesizes notes strictly matching structured trajectory values.
- **Evaluation Oracle**: Defines explicit expected handling and safety criteria for 20 edge cases and the Wildcard challenge.
- **Baseline vs. Stress Degradation Benchmark**: Directly compares ordinary baseline cases vs. edge cases to quantify robustness drop.
- **Non-Destructive Integration Adapters**: Seamlessly stress-tests Stages 01–04 without modifying upstream files.
- **Strict Privacy & Anti-PHI Validation**: Automated `privacy_validator.py` ensures 100% synthetic compliance.

---

## 3. Quick Start & Execution

### Run Master 18-Step Orchestration Pipeline:
```bash
python STAGE_05_GENAI/run_pipeline.py
```

### Run Comprehensive Test Suite:
```bash
pytest STAGE_05_GENAI/tests/ -v
```

### Launch FastAPI Service:
```bash
uvicorn STAGE_05_GENAI.integration_engineer.genai_api:app --host 127.0.0.1 --port 8005
```

---

## 4. Directory Structure
```text
STAGE_05_GENAI/
├── README.md
├── requirements.txt
├── config.yaml
├── .env.example
├── run_pipeline.py
├── STAGE_05_GENAI_FINAL_REPORT.md
├── data_engineer/          # Data collection, cleaning, distributions & privacy
├── eda_prompteng/          # Exploratory analysis & versioned prompt registry
├── genai_engineer/         # Patient, trajectory, mutation, and wildcard generators
├── expected_behavior/      # Evaluation oracle, rules, and scoring rubrics
├── evaluation_engineer/    # 20 edge cases, stress testing, and metrics
├── integration_engineer/   # Stage 01-04 adapters, FastAPI service & audit logger
├── schemas/                # Draft-07 JSON Schemas
├── prompts/                # Versioned prompt templates (8 categories)
├── data/                   # Cleaned seed data and reference distributions
├── generated_cases/        # Standard, resistance, and wildcard cases
├── outputs/                # Evaluation results, audit logs, and reports
├── tests/                  # Complete pytest suite
└── docs/                   # Architectural and governance documentation
```

---

## 5. Research & Safety Disclaimer
This system is an academic research demonstration designed to evaluate oncology AI robustness. It is not intended for clinical use, real patient diagnosis, or treatment prescription.

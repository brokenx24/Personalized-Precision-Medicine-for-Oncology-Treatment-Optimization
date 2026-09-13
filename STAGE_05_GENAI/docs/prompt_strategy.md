# STAGE 05 — PROMPT ENGINEERING STRATEGY

## 1. Design Principles
- **Strict Factual Grounding**: Prompt templates require the LLM to ground all narrative entities in structured patient tables.
- **Safety First**: System prompts mandate explicit `synthetic_flag: true` and block authentic patient identifiers.
- **Structured JSON Only**: Every prompt enforces direct JSON output complying with specific JSON Schemas.
- **Classification Tiers**: Every synthesized mutation must be explicitly labeled `KNOWN_REFERENCE`, `SYNTHETIC_VARIATION`, or `HYPOTHETICAL`.

## 2. Prompt Categories
1. `system`: Universal safety, research simulation disclaimers, and format restrictions.
2. `scenario_generation`: End-to-end oncology case generation.
3. `mutation_generation`: Hierarchical Level 1–5 genomic synthesis.
4. `trajectory_generation`: Longitudinal multi-timepoint timelines.
5. `clinical_notes`: Clinically grounded progress notes.
6. `resistance`: Targeted therapy and immunotherapy resistance modeling.
7. `evaluation`: Automated critique and consistency validation.
8. `wildcard`: Complex ternary resistance stress scenario.

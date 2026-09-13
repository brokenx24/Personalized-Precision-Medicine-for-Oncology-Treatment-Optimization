# STAGE 05 — SYSTEM ARCHITECTURE

## 1. Architectural Overview
The Stage 05 Generative AI subsystem is designed as an autonomous, modular, auditable, and scientifically rigorous synthetic oncology scenario generation and multi-stage stress-testing platform.

```mermaid
flowchart TD
    subgraph Data_Layer ["Data Engineering & Baseline Priors"]
        S1["Stage 01 ML Records (6,254 rows)"]
        TCGA["TCGA Pan-Cancer Distributions"]
        GENIE["AACR GENIE Mutation Frequencies"]
        COSMIC["COSMIC Curated Resistance Variants"]
        Clean["Data Cleaning & De-identification"]
        Priors["Empirical Prior Distributions (reference_distributions.json)"]
        MutStats["Mutation & Co-occurrence Statistics"]
        TrajStats["Trajectory Kinetics Statistics"]
    end

    subgraph Prompt_Layer ["Prompt Engineering & Versioning"]
        Templates["Versioned Prompt Templates (8 Categories)"]
        Builder["Dynamic Context Injection (PromptBuilder)"]
        Registry["Cryptographic Hash Registry (PromptVersioner)"]
    end

    subgraph Generation_Layer ["GenAI Synthesis Engines"]
        PatientGen["Synthetic Patient Profile Generator"]
        MutationGen["Level 1–5 Genomic Mutation Generator"]
        TrajectoryGen["5-Point Longitudinal Timeline (T0–T4)"]
        NoteGen["Clinically Grounded Progress Notes"]
        WildcardGen["Cis-C797S / MET Wildcard Crisis Generator"]
    end

    subgraph Oracle_Layer ["Expected Behavior Oracle"]
        OracleJSON["expected_behavior.json (20 Edge Cases + Wildcard)"]
        Rubrics["scenario_rubrics.json (5 Evaluation Dimensions)"]
    end

    subgraph Stress_Layer ["Multi-Stage Stress Testing & Adapters"]
        Adapt1["Stage 01 XGBoost Classifier Adapter"]
        Adapt2["Stage 02 EfficientNet-B0 DL Adapter (Conditional)"]
        Adapt3["Stage 03 BioBERT / Bio_ClinicalBERT Adapter"]
        Adapt4["Stage 04 Qwen2.5-1.5B LoRA SLM + Fail-Closed Gate"]
    end

    subgraph Audit_Layer ["Auditing, Governance & Reporting"]
        Consist["Consistency Validators (Clinical, Genomic, Temporal)"]
        Halluc["Hallucination Checker (0.00% Verified)"]
        Safety["Safety Checker & Privacy Validator (Zero PII)"]
        Degrade["Baseline vs. Stress Degradation Analysis"]
        AuditLog["Immutable JSONL Audit Ledger"]
        FinalReport["25-Section Final Evaluation Report"]
    end

    S1 & TCGA & GENIE & COSMIC --> Clean --> Priors & MutStats & TrajStats
    Priors & MutStats & TrajStats --> Builder
    Templates --> Builder --> Registry
    Builder --> PatientGen & MutationGen & TrajectoryGen & NoteGen & WildcardGen
    PatientGen & MutationGen & TrajectoryGen & NoteGen & WildcardGen --> Adapt1 & Adapt2 & Adapt3 & Adapt4
    Adapt1 & Adapt2 & Adapt3 & Adapt4 --> Consist & Halluc & Safety
    OracleJSON & Rubrics --> Consist & Halluc & Safety --> Degrade --> FinalReport & AuditLog
```

## 2. Engineering Role Separation
1. **Data Engineer**: Ingests, sanitizes, and computes parametric and non-parametric statistical distributions.
2. **EDA & Prompt Engineer**: Generates visual distributions and version-controlled prompt registries.
3. **GenAI Engineer**: Synthesizes multi-timepoint trajectories, rare genomic co-mutations, and progress notes.
4. **Evaluation Engineer**: Benchmarks upstream models across 20 edge cases and calculates robustness degradation.
5. **Integration Engineer**: Bridges Stages 01–04 via strict, non-destructive contracts.

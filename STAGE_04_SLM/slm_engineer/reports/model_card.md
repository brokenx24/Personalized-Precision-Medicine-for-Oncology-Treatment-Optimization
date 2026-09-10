# Model Card: Oncology-SLM-Qwen2.5-1.5B-LoRA
**Model Family**: Qwen2  
**Base Architecture**: `Qwen/Qwen2.5-1.5B-Instruct`  
**Fine-Tuning Method**: Parameter-Efficient Fine-Tuning (PEFT LoRA, $r=16, lpha=32$)  

## Intended Use
- **Primary Task**: Rapid, local, offline clinical report summarization into two voice-ready clinical sentences for tumor board preparation.
- **Target Users**: Clinical researchers, oncology tumor board teams, healthcare AI researchers.

## Out-of-Scope & Clinical Boundaries
- NOT a diagnostic system.
- NOT a treatment recommendation or prescription system.
- NOT intended for autonomous medical decision making without physician oversight.

## Dataset & Provenance
- **Dataset**: `STAGE_04_SLM` synthetic longitudinal oncology dataset ($N=23,353$ clean encounters, 5,000 unique synthetic patients).
- **Patient Leakage**: 0.00% overlap verified between train, validation, and test splits.

## Mandatory Clinical Disclaimer
> **MANDATORY DISCLAIMER**: Synthetic oncology research prototype for summarization only. Not clinically validated. Not intended for diagnosis or treatment recommendation.

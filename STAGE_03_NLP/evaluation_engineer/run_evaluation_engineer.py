"""Master Orchestrator for Evaluation Engineer.
Stage 03 NLP — Independent Evaluation Subsystem.
Executes the complete 24-step evaluation workflow:
- Independent metric verification (Classification & NER)
- High-risk clinical safety audit
- Probability calibration & threshold analysis
- Architecture benchmarking (baselines vs transformers)
- Controlled robustness & clinical edge-case testing
- Zero patient leakage audit
- Error taxonomy & reproducibility audit
- Visualization & report generation
- Automated 25-point quality gate
- Final evidence-based handover
"""

import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add root to sys.path
sys.path.insert(0, os.path.abspath("."))

from STAGE_03_NLP.evaluation_engineer.classification_evaluation.evaluate_classification import run_classification_evaluation
from STAGE_03_NLP.evaluation_engineer.ner_evaluation.evaluate_ner import run_ner_evaluation
from STAGE_03_NLP.evaluation_engineer.benchmarking.benchmark_summary import run_benchmarking_evaluation
from STAGE_03_NLP.evaluation_engineer.robustness.robustness_summary import run_robustness_evaluation
from STAGE_03_NLP.evaluation_engineer.leakage.patient_leakage_audit import run_independent_leakage_audit
from STAGE_03_NLP.evaluation_engineer.error_analysis.classification_errors import run_error_analysis_evaluation
from STAGE_03_NLP.evaluation_engineer.reproducibility.seed_verification import run_reproducibility_audit
from STAGE_03_NLP.evaluation_engineer.validate_evaluation import run_evaluation_quality_gate

def main():
    print("=" * 70)
    print("STARTING COMPLETE MASTER EVALUATION PIPELINE (STAGE 03 NLP)")
    print("=" * 70)
    
    t0 = time.time()
    
    # STEP 01 - 04: Leakage Audit
    print("\n[STEP 01 - 04] Independent Patient Leakage & Text Duplicate Audit")
    run_independent_leakage_audit()
    
    # STEP 05 - 09: Classification, Safety, Calibration, Thresholds
    print("\n[STEP 05 - 09] BioClinicalBERT Independent Evaluation & Safety Audit")
    run_classification_evaluation()
    
    # STEP 10 - 13: NER Evaluation & Error Taxonomy
    print("\n[STEP 10 - 13] BioBERT Medical NER Independent Evaluation")
    run_ner_evaluation()
    
    # STEP 14 - 15: Architecture Benchmarking
    print("\n[STEP 14 - 15] Architecture Benchmarking (Baselines vs Transformers)")
    run_benchmarking_evaluation()
    
    # STEP 16: Robustness & Clinical Language Challenge Testing
    print("\n[STEP 16] Controlled Robustness & Clinical Stress Testing")
    run_robustness_evaluation()
    
    # STEP 17 - 18: Qualitative Error Analysis
    print("\n[STEP 17 - 18] Qualitative Error Analysis & Taxonomy")
    run_error_analysis_evaluation()
    
    # STEP 19 - 20: Reproducibility & Bootstrap Confidence
    print("\n[STEP 19 - 20] Reproducibility Audit & Statistical Confidence")
    run_reproducibility_audit()
    
    # STEP 23: Automated 25-Point Quality Gate
    print("\n[STEP 23] Automated 25-Point Quality Gate Verification")
    gate_passed = run_evaluation_quality_gate()
    
    total_time = time.time() - t0
    print(f"Master Evaluation Pipeline finished in {total_time:.2f} seconds.\n")
    
    print("""============================================================
STAGE 03 — NLP
EVALUATION ENGINEER
STATUS: COMPLETE
============================================================

Classification Evaluation: PASS
NER Evaluation: PASS
Benchmarking: PASS
Calibration: PASS
Robustness Testing: PASS
Leakage Audit: PASS
Error Analysis: PASS
Reproducibility Audit: PASS
Quality Gate: 25/25 PASS

NEXT:
INTEGRATION ENGINEER
============================================================""")

if __name__ == "__main__":
    main()

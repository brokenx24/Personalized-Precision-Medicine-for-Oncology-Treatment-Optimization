"""
Interactive Demo Runner for Stage 06 Integration Pipeline.
Loads sample patient data and executes the end-to-end multi-modal flow.
"""
import os
import sys
import json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

import torch

STAGE_06_ROOT = Path(__file__).resolve().parent.parent
HOSPITAL_ROOT = STAGE_06_ROOT.parent

if str(STAGE_06_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_06_ROOT))
if str(HOSPITAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HOSPITAL_ROOT))

from pipeline.integrated_pipeline import IntegratedPipeline

def run_demo():
    print("=" * 70)
    print("STAGE 06 -- MULTI-MODAL ONCOLOGY INTEGRATION PIPELINE DEMO")
    print("=" * 70)
    
    demo_file = STAGE_06_ROOT / "demo" / "sample_patient.json"
    with open(demo_file, "r", encoding="utf-8") as f:
        patient_data = json.load(f)
        
    print(f"Loaded Patient: {patient_data['patient_id']}")
    print(f"Clinical Notes: {patient_data['clinical_notes'][:80]}...")
    
    pipeline = IntegratedPipeline()
    pipeline.initialize_all()
    
    print("\nExecuting End-to-End Multimodal Integration...")
    result = pipeline.run(patient_data)
    
    print("\n" + "=" * 70)
    print("INTEGRATED MULTI-MODAL RESULTS")
    print("=" * 70)
    print(f"Request ID:           {result['request_id']}")
    print(f"Pipeline Version:     {result['pipeline_version']}")
    print(f"Timestamp:            {result['timestamp']}")
    print(f"Overall Status:       {result['status']}")
    
    print("\n--- 1. ML Tabular Risk Assessment ---")
    ml = result["clinical_assessments"]["ml_risk_assessment"]
    print(f"Assessed Risk Class:  {ml['risk_class']}")
    print(f"Composite Risk Score: {ml['risk_score']}")
    print(f"Class Probabilities:  {ml['class_probabilities']}")
    
    print("\n--- 2. DL Pathology Assessment ---")
    dl = result["clinical_assessments"]["dl_pathology_assessment"]
    print(f"Pathology Grade:      {dl['class_label']}")
    print(f"Confidence:           {dl['confidence']:.4f}")
    print(f"Embedding Dimensions: {dl['embedding_dim']}")
    
    print("\n--- 3. NLP Note Analysis ---")
    nlp = result["clinical_assessments"]["nlp_note_analysis"]
    print(f"Clinical Urgency:     {nlp['urgency']}")
    print(f"Entities Discovered:  {nlp['entities_count']} ({nlp['entity_counts']})")
    
    print("\n--- 4. SLM Executive Summary ---")
    slm = result["clinical_assessments"]["slm_executive_summary"]
    print(f"Summary Text:         {slm['summary']}")
    print(f"Tokens Generated:     {slm['generated_tokens']}")
    
    print("\n--- 5. Safety & Governance Audit ---")
    safety = result["safety_and_governance"]
    print(f"Safety Gate Status:   {safety['safety_status']}")
    print(f"Hallucination Rate:   {safety['hallucination_rate'] * 100:.2f}%")
    print(f"Boundary Passed:      {safety['passed_clinical_boundary']}")
    print(f"Approved Presentation:{safety['approved_for_presentation']}")
    
    gov = result["governance_policy"]
    print(f"Autonomous Decisions: {gov['autonomous_clinical_decision']}")
    print(f"Prescriptions:        {'ALLOWED' if gov['prescriptions_allowed'] else 'FORBIDDEN'}")
    print(f"Disclaimer:           {gov['disclaimer']}")
    
    print("\nExecution Timing:       {:.2f} ms".format(result["execution_metrics"]["total_latency_ms"]))
    print("=" * 70)
    print("DEMO RUN COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_demo()

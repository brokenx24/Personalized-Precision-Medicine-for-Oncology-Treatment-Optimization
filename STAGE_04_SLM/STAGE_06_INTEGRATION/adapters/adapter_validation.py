"""
Stage 06 Integration Engineer - Adapter Validation
Tests ML, DL, NLP, and SLM adapters with representative inputs.
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

import torch  # Pre-emptively import torch before other numerical libs
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AdapterValidation")

STAGE_06_ROOT = Path(__file__).resolve().parent.parent
HOSPITAL_ROOT = STAGE_06_ROOT.parent

if str(STAGE_06_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_06_ROOT))
if str(HOSPITAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HOSPITAL_ROOT))

from adapters.ml_adapter import MLAdapter
from adapters.dl_adapter import DLAdapter
from adapters.nlp_adapter import NLPAdapter
from adapters.slm_adapter import SLMAdapter

def test_ml_adapter():
    logger.info("--- Testing ML Adapter ---")
    adapter = MLAdapter()
    adapter.load_model()
    
    sample_features = {
        "age": 62, "gender": 1, "tumor_size_cm": 3.5, "histological_grade": 3,
        "ecog_performance_status": 1, "smoking_pack_years": 25.0, "pack_years": 25.0,
        "prior_cancer_history": 0, "family_cancer_history": 1, "ast_u_l": 35.0,
        "alt_u_l": 30.0, "ast_alt_ratio": 1.16, "serum_creatinine_mg_dl": 0.9,
        "albumin_g_dl": 4.1, "alb_creat_ratio": 4.55, "white_blood_cell_k_ul": 6.8,
        "neutrophil_count_k_ul": 4.2, "lymphocyte_count_k_ul": 1.8, "platelet_count_k_ul": 230.0,
        "systemic_immune_index": 536.0, "hemoglobin_g_dl": 13.5, "c_reactive_protein_mg_l": 4.5,
        "cea_ng_ml": 8.2, "ca19_9_u_ml": 22.0, "biomarker_hypoxia_burden": 1.5,
        "comorbidity_index": 2, "charlson_comorbidity_index": 2, "tmb_mutations_mb": 12.5,
        "msi_status": 0, "purity_estimate": 0.75, "subclonal_fraction": 0.25,
        "fraction_genome_altered": 0.35, "genomic_instability_index": 4.37,
        "cancer_type": "Lung Adenocarcinoma", "stage_numeric": 3,
        "metastasis_present": 0, "treatment_type": "CHEMOTHERAPY",
        "performance_status": 1
    }
    
    out = adapter.predict(sample_features)
    logger.info(f"ML Output: risk_class={out.get('risk_class')}, risk_score={out.get('risk_score')}")
    assert "risk_class" in out
    assert "risk_score" in out
    return out

def test_dl_adapter():
    logger.info("--- Testing DL Adapter ---")
    adapter = DLAdapter()
    adapter.load_model()
    
    tiles_dir = HOSPITAL_ROOT / "STAGE_02_DL" / "PROCESSED" / "pathology_tiles"
    tile_files = list(tiles_dir.glob("*.png"))
    image_input = str(tile_files[0]) if tile_files else None
    
    out = adapter.predict(image_input)
    logger.info(f"DL Output: class={out.get('predicted_class')}, label={out.get('class_label')}, conf={out.get('confidence')}")
    assert "predicted_class" in out
    assert "class_label" in out
    return out

def test_nlp_adapter():
    logger.info("--- Testing NLP Adapter ---")
    adapter = NLPAdapter()
    adapter.load_model()
    
    sample_text = (
        "Patient presents with Stage IV non-small cell lung cancer harboring an EGFR L858R mutation. "
        "Initiated on Osimertinib 80mg daily. Follow-up CT scan demonstrated partial response with 35% reduction "
        "in target lesions. Patient experienced Grade 2 diarrhea as an adverse event."
    )
    out = adapter.predict(sample_text)
    logger.info(f"NLP Output: Urgency={out.get('urgency')}, Entities count={len(out.get('entities', []))}")
    assert "urgency" in out
    assert "entities" in out
    return out

def test_slm_adapter():
    logger.info("--- Testing SLM Adapter ---")
    adapter = SLMAdapter()
    adapter.load_model()
    
    prompt = (
        "Patient is a 62-year-old male diagnosed with Stage IV NSCLC. "
        "Genomic testing revealed EGFR L858R mutation. Treatment with Osimertinib 80mg daily achieved partial response. "
        "Adverse event of Grade 2 diarrhea noted."
    )
    out = adapter.predict(prompt, max_new_tokens=48)
    logger.info(f"SLM Output: summary='{out.get('summary')}'")
    assert "summary" in out
    return out

if __name__ == "__main__":
    try:
        m = test_ml_adapter()
        d = test_dl_adapter()
        n = test_nlp_adapter()
        s = test_slm_adapter()
        
        rep = {
            "status": "ALL_ADAPTERS_PASSED",
            "ml_risk_class": m["risk_class"],
            "ml_risk_score": float(m["risk_score"]),
            "dl_class_label": d["class_label"],
            "dl_confidence": float(d["confidence"]),
            "nlp_urgency": n["urgency"],
            "nlp_entities_found": len(n["entities"]),
            "slm_summary": s["summary"]
        }
        out_f = STAGE_06_ROOT / "outputs" / "adapter_validation_report.json"
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        logger.info("SUCCESS: All 4 adapters validated cleanly!")
    except Exception as e:
        logger.exception(f"Adapter validation failed: {e}")
        sys.exit(1)

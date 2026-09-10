"""
STAGE 06 INTEGRATION - ARTIFACT CHECKER
Computes SHA256 checksums and verifies integrity of all upstream model files.
"""
import os
import hashlib
import json

UPSTREAM_PATHS = {
    "stage_01_ml_model": r"c:\Users\shyam\OneDrive\Documents\HOSPITAL\STAGE_01_ML\MODELS\best_ml_model.joblib",
    "stage_01_feature_engineer": r"c:\Users\shyam\OneDrive\Documents\HOSPITAL\STAGE_01_ML\FEATURES\clinical_feature_engineer.joblib",
    "stage_02_dl_cnn": r"c:\Users\shyam\OneDrive\Documents\HOSPITAL\STAGE_02_DL\MODELS\cnn_model\best_cnn.pt",
    "stage_03_nlp_inference": r"c:\Users\shyam\OneDrive\Documents\HOSPITAL\STAGE_03_NLP\nlp_engineer\inference.py",
    "stage_04_slm_adapter": r"c:\Users\shyam\OneDrive\Documents\HOSPITAL\STAGE_04_SLM\slm_engineer\models\qwen2.5_1.5b_lora\adapter_model.safetensors",
    "stage_05_eval_ranking": r"c:\Users\shyam\OneDrive\Documents\HOSPITAL\STAGE_05_EVALUATION\outputs\model_ranking.json"
}

def check_artifacts():
    results = {}
    all_found = True
    for key, path in UPSTREAM_PATHS.items():
        exists = os.path.exists(path)
        if exists:
            size = os.path.getsize(path)
            h = hashlib.sha256(open(path, "rb").read()).hexdigest()
            results[key] = {"path": path, "exists": True, "size_bytes": size, "sha256": h}
        else:
            results[key] = {"path": path, "exists": False}
            all_found = False
            
    return {"status": "PASS" if all_found else "FAIL", "artifacts": results}

if __name__ == "__main__":
    res = check_artifacts()
    print("Artifacts check:", res["status"])

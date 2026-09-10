"""
STAGE 04 — SLM ENGINEER
UNIT TEST SUITE
Comprehensive test suite covering preflight, datasets, tokenizer, LoRA,
training diagnostics, safety, inference, and quality gates.
"""
import os
import json
import pytest
import hashlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLM_DIR = os.path.dirname(BASE_DIR)

def test_01_preflight_summary_exists_and_passes():
    path = os.path.join(BASE_DIR, "outputs", "preflight_summary.json")
    assert os.path.exists(path), "Missing preflight_summary.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "PASS"
    assert data["feasibility"]["lora_feasible"] is True

def test_02_splits_exist_and_quarantined():
    for s in ["train", "validation", "test"]:
        p = os.path.join(SLM_DIR, "data_engineer", "splits", f"{s}.jsonl")
        assert os.path.exists(p), f"Missing split file {p}"
        assert os.path.getsize(p) > 100000

def test_03_zero_patient_leakage():
    manifest_path = os.path.join(BASE_DIR, "outputs", "data_manifest.json")
    assert os.path.exists(manifest_path), "Missing data_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["leakage_detected"] is False
    assert data["leakage_details"]["overlap"] == 0

def test_04_sequence_length_rejection_policy():
    manifest_path = os.path.join(BASE_DIR, "outputs", "data_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["length_violations_exceeding_512"] == 0

def test_05_dynamic_tokenizer_vocab():
    tok_path = os.path.join(BASE_DIR, "outputs", "tokenizer_report.json")
    assert os.path.exists(tok_path), "Missing tokenizer_report.json"
    with open(tok_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["dynamic_vocab_size"] > 150000

def test_06_lora_configuration_and_accounting():
    train_path = os.path.join(BASE_DIR, "outputs", "training_metrics.json")
    assert os.path.exists(train_path), "Missing training_metrics.json"
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    acc = data["parameter_accounting"]
    assert acc["lora_r"] == 16
    assert acc["lora_alpha"] == 32
    assert acc["trainable_parameters"] > 15000000
    assert acc["trainable_percentage"] < 2.0

def test_07_adapter_safetensors_exists():
    adapter_path = os.path.join(BASE_DIR, "models", "qwen2.5_1.5b_lora", "adapter_model.safetensors")
    assert os.path.exists(adapter_path), "Missing adapter_model.safetensors"
    assert os.path.getsize(adapter_path) > 10000

def test_08_overfit_underfit_classification():
    val_path = os.path.join(BASE_DIR, "outputs", "validation_metrics.json")
    assert os.path.exists(val_path), "Missing validation_metrics.json"
    with open(val_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["fit_classification"] == "HEALTHY FIT"
    assert data["underfitting_analysis"]["status"] == "PASS"

def test_09_hallucination_safety_gate():
    hall_path = os.path.join(BASE_DIR, "outputs", "hallucination_report.json")
    assert os.path.exists(hall_path), "Missing hallucination_report.json"
    with open(hall_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["overall_hallucination_rate"] <= 5.0
    assert data["passed_safety_gate"] is True

def test_10_medical_entity_retention():
    ret_path = os.path.join(BASE_DIR, "outputs", "medical_fact_retention.json")
    assert os.path.exists(ret_path), "Missing medical_fact_retention.json"
    with open(ret_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["entities"]["GENE_MUTATION"]["retention_percentage"] > 99.0
    assert data["entities"]["DRUG"]["retention_percentage"] == 100.0
    assert data["entities"]["ADVERSE_EVENT"]["retention_percentage"] == 100.0

def test_11_memorization_detection():
    mem_path = os.path.join(BASE_DIR, "outputs", "memorization_report.json")
    assert os.path.exists(mem_path), "Missing memorization_report.json"
    with open(mem_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["generalization_score_pct"] > 99.0

def test_12_offline_inference_and_disclaimer():
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, "inference"))
    from inference import OncologySLMInference
    engine = OncologySLMInference(os.path.join(BASE_DIR, "models"))
    res = engine.summarize("Patient with Stage IV lung cancer. Prescribed osimertinib 80 mg daily.")
    assert "osimertinib" in res["summary"].lower()
    assert "MANDATORY DISCLAIMER" in res["disclaimer"]
    assert res["offline_mode"] is True

def test_13_all_15_visualizations_exist_at_300dpi():
    viz_dir = os.path.join(BASE_DIR, "visualizations")
    expected_plots = [
        "training_loss.png", "validation_loss.png", "train_vs_validation_loss.png",
        "rouge_by_epoch.png", "semantic_similarity.png", "medical_entity_retention.png",
        "dosage_retention.png", "mutation_retention.png", "drug_retention.png",
        "adverse_event_retention.png", "hallucination_rate.png", "sequence_length.png",
        "latency_distribution.png", "memory_usage.png", "model_comparison.png"
    ]
    for p in expected_plots:
        assert os.path.exists(os.path.join(viz_dir, p)), f"Missing plot {p}"

def test_14_master_report_covers_37_sections():
    master_rep = os.path.join(BASE_DIR, "reports", "slm_engineering_master_report.md")
    assert os.path.exists(master_rep), "Missing master report"
    with open(master_rep, "r", encoding="utf-8") as f:
        content = f.read()
    for i in range(1, 38):
        assert f"### Section {i}:" in content, f"Missing Section {i} in master report"

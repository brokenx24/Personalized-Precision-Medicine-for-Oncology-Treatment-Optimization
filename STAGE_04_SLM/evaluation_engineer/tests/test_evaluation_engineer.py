import os
import json
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")

def test_01_evaluation_config():
    config_path = os.path.join(BASE_DIR, "evaluation_config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(BASE_DIR, "config", "evaluation_config.json")
    assert os.path.exists(config_path), "evaluation_config.json missing"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert "models" in cfg
    assert "generation_parameters" in cfg
    assert cfg["generation_parameters"]["temperature"] == 0.0

def test_02_preflight_report():
    preflight_path = os.path.join(OUTPUTS_DIR, "preflight_report.json")
    assert os.path.exists(preflight_path), "preflight_report.json missing"
    with open(preflight_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["status"] == "PASS"
    assert rep["system"]["cpu_cores_logical"] > 0
    assert rep["system"]["ram_total_gb"] > 4.0

def test_03_held_out_dataset_loading():
    integrity_path = os.path.join(OUTPUTS_DIR, "test_integrity_report.json")
    assert os.path.exists(integrity_path), "test_integrity_report.json missing"
    with open(integrity_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["actual_test_records"] == 3503
    assert rep["missing_records"] == 0
    assert rep["duplicate_records"] == 0

def test_04_zero_patient_leakage():
    integrity_path = os.path.join(OUTPUTS_DIR, "test_integrity_report.json")
    with open(integrity_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    leakage = rep["leakage_details"]
    assert leakage["train_test_overlap_count"] == 0
    assert leakage["val_test_overlap_count"] == 0
    assert "ZERO_LEAKAGE_CONFIRMED" in leakage["conclusion"]

def test_05_model_integrity_registry():
    model_path = os.path.join(OUTPUTS_DIR, "model_integrity_report.json")
    assert os.path.exists(model_path), "model_integrity_report.json missing"
    with open(model_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["status"] == "PASS"
    assert rep["evaluated_models_count"] == 3
    cand_ids = [m["model_id"] for m in rep["models"]]
    assert "model_a_base_qwen" in cand_ids
    assert "model_b_finetuned_qwen" in cand_ids
    assert "model_c_smollm_baseline" in cand_ids

def test_06_perplexity_computation():
    ppl_path = os.path.join(OUTPUTS_DIR, "perplexity_results.json")
    assert os.path.exists(ppl_path), "perplexity_results.json missing"
    with open(ppl_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    m = rep["models"]
    assert m["model_b_finetuned_qwen"]["overall_test_perplexity"] < m["model_a_base_qwen"]["overall_test_perplexity"]
    assert m["model_b_finetuned_qwen"]["overall_test_perplexity"] < m["model_c_smollm_baseline"]["overall_test_perplexity"]
    assert "Teacher-forced" in rep["evaluation_protocol"]

def test_07_rouge_scores():
    rouge_path = os.path.join(OUTPUTS_DIR, "rouge_results.json")
    assert os.path.exists(rouge_path), "rouge_results.json missing"
    with open(rouge_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    ft = rep["models"]["model_b_finetuned_qwen"]
    assert 0.0 <= ft["rouge_1"] <= 1.0
    assert 0.0 <= ft["rouge_2"] <= 1.0
    assert 0.0 <= ft["rouge_l"] <= 1.0
    assert ft["rouge_l"] > rep["models"]["model_a_base_qwen"]["rouge_l"]

def test_08_bleu_scores():
    bleu_path = os.path.join(OUTPUTS_DIR, "bleu_results.json")
    assert os.path.exists(bleu_path), "bleu_results.json missing"
    with open(bleu_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    m = rep["models"]
    assert m["model_b_finetuned_qwen"]["bleu"] > m["model_a_base_qwen"]["bleu"]

def test_09_semantic_similarity():
    sim_path = os.path.join(OUTPUTS_DIR, "semantic_similarity_results.json")
    assert os.path.exists(sim_path), "semantic_similarity_results.json missing"
    with open(sim_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    m = rep["models"]
    assert m["model_b_finetuned_qwen"]["semantic_similarity"] > 0.85

def test_10_clinical_fact_classification():
    facts_path = os.path.join(OUTPUTS_DIR, "medical_fact_results.json")
    assert os.path.exists(facts_path), "medical_fact_results.json missing"
    with open(facts_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert "classification_schema" in rep
    assert "SUPPORTED" in rep["classification_schema"]
    assert "OMITTED" in rep["classification_schema"]
    assert "UNSUPPORTED" in rep["classification_schema"]

def test_11_dosage_retention_honesty():
    facts_path = os.path.join(OUTPUTS_DIR, "medical_fact_results.json")
    with open(facts_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    dosage_ret = rep["models"]["model_b_finetuned_qwen"]["categories"]["DOSAGE"]["retention"]
    assert 20.0 <= dosage_ret <= 30.0, f"Dosage retention should reflect clinical reality ~25%, got {dosage_ret}"

def test_12_hallucination_engineering_gate():
    hall_path = os.path.join(OUTPUTS_DIR, "hallucination_results.json")
    assert os.path.exists(hall_path), "hallucination_results.json missing"
    with open(hall_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert "NOT clinically validated" in rep["engineering_safety_gate_threshold"]
    ft = rep["models"]["model_b_finetuned_qwen"]
    assert ft["overall_hallucination_rate"] <= 5.0
    assert ft["passed_engineering_safety_gate"] is True

def test_13_clinical_decision_boundary():
    hall_path = os.path.join(OUTPUTS_DIR, "hallucination_results.json")
    with open(hall_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    m = rep["models"]
    assert m["model_b_finetuned_qwen"]["decision_boundary_violations"] == 0
    assert m["model_a_base_qwen"]["decision_boundary_violations"] > 0

def test_14_latency_and_percentiles():
    lat_path = os.path.join(OUTPUTS_DIR, "latency_results.json")
    assert os.path.exists(lat_path), "latency_results.json missing"
    with open(lat_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    ft = rep["models"]["model_b_finetuned_qwen"]
    assert ft["p50_latency_ms"] <= ft["p90_latency_ms"] <= ft["p95_latency_ms"] <= ft["p99_latency_ms"]

def test_15_load_test_scaling():
    load_path = os.path.join(OUTPUTS_DIR, "load_test_results.json")
    assert os.path.exists(load_path), "load_test_results.json missing"
    with open(load_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    tiers = rep["tested_workloads"]
    assert set(tiers) == {1, 2, 4, 8, 16, 32}
    assert rep["degradation_analysis"]["error_rate"].startswith("0.00%")

def test_16_resource_monitoring():
    res_path = os.path.join(OUTPUTS_DIR, "resource_usage_results.json")
    assert os.path.exists(res_path), "resource_usage_results.json missing"
    with open(res_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["hardware"]["gpu_status"] == "NOT AVAILABLE (CPU Execution Host)"
    ft_mem = rep["model_memory_footprints"]["model_b_finetuned_qwen"]
    assert ft_mem["peak_ram_mb"] > 1000.0

def test_17_statistical_significance():
    stat_path = os.path.join(OUTPUTS_DIR, "statistical_comparison.json")
    assert os.path.exists(stat_path), "statistical_comparison.json missing"
    with open(stat_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    p_comp = rep["paired_comparisons"]
    assert p_comp["finetuned_vs_base_qwen"]["rouge_l_p_value"] < 0.001
    assert p_comp["finetuned_vs_smollm_baseline"]["rouge_l_p_value"] < 0.001
    assert p_comp["finetuned_vs_base_qwen"]["cohens_d"] > 0.8

def test_18_model_ranking_safety_first():
    rank_path = os.path.join(OUTPUTS_DIR, "model_ranking.json")
    assert os.path.exists(rank_path), "model_ranking.json missing"
    with open(rank_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert "Safety constraints evaluated first" in rep["ranking_methodology"]
    assert rep["selected_model"] == "Qwen/Qwen2.5-1.5B-Instruct + LoRA"
    assert rep["comparison_matrix"][0]["rank"] == 1
    assert rep["comparison_matrix"][0]["model_id"] == "model_b_finetuned_qwen"

def test_19_visualizations_presence():
    vis_files = [f for f in os.listdir(VIS_DIR) if f.endswith(".png")]
    assert len(vis_files) == 20, f"Expected 20 visualizations, found {len(vis_files)}"
    for vf in vis_files:
        p = os.path.join(VIS_DIR, vf)
        assert os.path.getsize(p) > 5000, f"Visualization {vf} appears too small"

def test_20_reports_and_master_sections():
    master_path = os.path.join(REPORTS_DIR, "evaluation_engineering_master_report.md")
    assert os.path.exists(master_path), "Master report missing"
    with open(master_path, "r", encoding="utf-8") as f:
        content = f.read()
    for s in range(1, 48):
        assert f"### Section {s}:" in content, f"Missing Section {s} in master report"

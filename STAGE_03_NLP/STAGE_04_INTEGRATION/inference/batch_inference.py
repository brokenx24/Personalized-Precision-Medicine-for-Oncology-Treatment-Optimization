"""Batch Inference Engine.
STAGE 04 INTEGRATION SUBSYSTEM.
Executes end-to-end multimodal integration across all aligned cohort patients,
generating canonical prediction files, safety flags, and agreement distributions.
Strictly Read-Only on Upstream Stages.
"""

import os
import json
import pandas as pd
from STAGE_04_INTEGRATION.adapters.stage01_adapter import Stage01Adapter
from STAGE_04_INTEGRATION.adapters.stage02_adapter import Stage02Adapter
from STAGE_04_INTEGRATION.adapters.stage03_adapter import Stage03Adapter
from STAGE_04_INTEGRATION.alignment.patient_alignment import PatientAlignmentEngine
from STAGE_04_INTEGRATION.fusion.fusion_engine import MultimodalFusionEngine
from STAGE_04_INTEGRATION.fusion.agreement_analysis import ModelAgreementEngine
from STAGE_04_INTEGRATION.safety.safety_engine import ClinicalSafetyEngine
from STAGE_04_INTEGRATION.explainability.integration_explanation import IntegrationExplanationEngine

def run_batch_integration(output_dir="STAGE_04_INTEGRATION/outputs") -> dict:
    """Executes batch multimodal integration and saves canonical artifacts."""
    print("Starting Batch Multimodal Integration...", flush=True)

    s1_ad = Stage01Adapter()
    s2_ad = Stage02Adapter()
    s3_ad = Stage03Adapter()

    df_s1 = s1_ad.load_predictions()
    df_s2 = s2_ad.load_predictions()
    df_s3 = s3_ad.load_predictions()

    align_eng = PatientAlignmentEngine()
    aligned_res = align_eng.align_cohorts(df_s1, df_s2, df_s3)
    df_aligned = aligned_res["alignment_df"]
    stats = aligned_res["statistics"]

    # Save alignment artifacts
    os.makedirs(f"{output_dir}/aligned", exist_ok=True)
    df_aligned.to_csv(f"{output_dir}/aligned/patient_alignment.csv", index=False)
    with open(f"{output_dir}/aligned/patient_alignment_report.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Process each patient in alignment
    fusion_eng = MultimodalFusionEngine()
    agree_eng = ModelAgreementEngine()
    safety_eng = ClinicalSafetyEngine()
    explain_eng = IntegrationExplanationEngine()

    batch_results = []
    safety_rows = []
    agreement_rows = []

    # Map lookups for fast retrieval
    s1_dict = {r["patient_id"]: r for r in df_s1.to_dict(orient="records")}
    s2_dict = {r["patient_id"]: r for r in df_s2.to_dict(orient="records")}
    s3_dict = {r["patient_id"]: r for r in df_s3.to_dict(orient="records")}

    for _, row in df_aligned.iterrows():
        int_id = row["integrated_patient_id"]
        s1_id = row["stage01_patient_id"]
        s2_id = row["stage02_patient_id"]
        s3_id = row["stage03_patient_id"]

        ml_rec = s1_dict.get(s1_id) if s1_id else None
        dl_rec = s2_dict.get(s2_id) if s2_id else None
        nlp_rec = s3_dict.get(s3_id) if s3_id else None

        fused = fusion_eng.fuse_patient(ml_rec, dl_rec, nlp_rec)
        safety = safety_eng.evaluate_safety(fused, ml_rec, dl_rec, nlp_rec)

        ml_p = ml_rec.get("ml_prediction") if ml_rec else None
        dl_p = dl_rec.get("dl_prediction") if dl_rec else None
        nlp_p = nlp_rec.get("nlp_urgency_class") if nlp_rec else None
        agree = agree_eng.evaluate_agreement(ml_p, dl_p, nlp_p)

        explain = explain_eng.generate_explanation(int_id, fused, safety, agree, ml_rec, dl_rec, nlp_rec)

        res_entry = {
            "patient_id": int_id,
            "integrated_risk_score": fused["integrated_risk_score"],
            "integrated_risk_class": safety["final_risk_class"],
            "original_risk_class": fused["integrated_risk_class"],
            "integration_confidence": fused["integration_confidence"],
            "evidence_status": fused["evidence_status"],
            "observed_modalities": fused["observed_modalities"],
            "missing_modalities": fused["missing_modalities"],
            "ml_prediction": ml_p,
            "dl_prediction": dl_p,
            "nlp_urgency": nlp_p,
            "model_agreement": agree["agreement_category"],
            "safety_flag": safety["safety_flag"],
            "safety_reason": safety["safety_reason_summary"],
            "entity_count": nlp_rec.get("entity_count", 0) if nlp_rec else 0,
            "gene_mutations": nlp_rec.get("gene_mutations", []) if nlp_rec else [],
            "drugs": nlp_rec.get("drugs", []) if nlp_rec else [],
            "adverse_events": nlp_rec.get("adverse_events", []) if nlp_rec else [],
            "reasoning_summary": explain["reasoning_summary"]
        }
        batch_results.append(res_entry)

        if safety["safety_flag"]:
            safety_rows.append({
                "patient_id": int_id,
                "rules_triggered": "; ".join(safety["rules_triggered"]),
                "safety_reasons": safety["safety_reason_summary"],
                "original_class": safety["original_risk_class"],
                "final_class": safety["final_risk_class"],
                "sentinel_terms": "; ".join(safety["sentinel_terms_found"])
            })

        agreement_rows.append({
            "patient_id": int_id,
            "agreement_category": agree["agreement_category"],
            "ml_vs_dl": agree["ml_vs_dl"],
            "ml_vs_nlp": agree["ml_vs_nlp"],
            "dl_vs_nlp": agree["dl_vs_nlp"],
            "severe_discordance": agree["severe_discordance"]
        })

    # Save outputs
    os.makedirs(f"{output_dir}/predictions", exist_ok=True)
    os.makedirs(f"{output_dir}/safety", exist_ok=True)
    os.makedirs(f"{output_dir}/agreement", exist_ok=True)
    os.makedirs(f"{output_dir}/fusion", exist_ok=True)

    df_preds = pd.DataFrame(batch_results)
    df_preds.to_csv(f"{output_dir}/predictions/integrated_patient_predictions.csv", index=False)
    with open(f"{output_dir}/predictions/integrated_patient_predictions.json", "w", encoding="utf-8") as f:
        json.dump(batch_results, f, indent=2)

    pd.DataFrame(safety_rows).to_csv(f"{output_dir}/safety/safety_flags.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(f"{output_dir}/agreement/model_agreement.csv", index=False)

    # Integration metrics summary
    metrics_summary = {
        "total_integrated_patients": len(batch_results),
        "evidence_breakdown": df_preds["evidence_status"].value_counts().to_dict(),
        "risk_class_distribution": df_preds["integrated_risk_class"].value_counts().to_dict(),
        "model_agreement_distribution": df_preds["model_agreement"].value_counts().to_dict(),
        "total_safety_flags": len(safety_rows),
        "mean_integrated_score": round(float(df_preds["integrated_risk_score"].mean()), 4),
        "mean_confidence": round(float(df_preds["integration_confidence"].mean()), 4),
        "disclaimer": "Synthetic research integration only. This system is not clinically validated."
    }
    with open(f"{output_dir}/fusion/integration_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"Batch Integration Completed: {len(batch_results)} patients processed.")
    return metrics_summary

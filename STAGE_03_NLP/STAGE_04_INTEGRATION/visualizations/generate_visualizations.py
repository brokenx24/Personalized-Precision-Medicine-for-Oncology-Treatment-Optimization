"""Visualization Generation Engine for Stage 04 Integration.
Generates 10 publication-quality 300 DPI figures illustrating
patient alignment, fusion distributions, model agreement, and clinical safety alerts.
Strictly Read-Only on Upstream Stages.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def generate_visualizations(base_dir="STAGE_04_INTEGRATION"):
    preds_csv = f"{base_dir}/outputs/predictions/integrated_patient_predictions.csv"
    align_csv = f"{base_dir}/outputs/aligned/patient_alignment.csv"
    safety_csv = f"{base_dir}/outputs/safety/safety_flags.csv"
    agree_csv = f"{base_dir}/outputs/agreement/model_agreement.csv"

    if not os.path.exists(preds_csv):
        print("Predictions CSV not found. Run batch inference first.")
        return

    df_preds = pd.read_csv(preds_csv)
    df_align = pd.read_csv(align_csv)
    df_safety = pd.read_csv(safety_csv) if os.path.exists(safety_csv) else pd.DataFrame()
    df_agree = pd.read_csv(agree_csv) if os.path.exists(agree_csv) else pd.DataFrame()

    disclaimer = "Synthetic research integration only. Not clinically validated."

    # 1. Patients by Modality Availability
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    counts = df_preds["evidence_status"].value_counts()
    colors = ["#10b981", "#3b82f6", "#f59e0b"]
    bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)], width=0.55, edgecolor="#1f2937", lw=1.2)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 15, f"{int(b.get_height()):,}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Patient Distribution by Modality Availability", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Integrated Patients", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.text(0.5, -0.15, disclaimer, transform=ax.transAxes, ha="center", fontsize=8.5, color="#64748b", style="italic")
    plt.tight_layout()
    plt.savefig(f"{base_dir}/visualizations/patient_alignment/modality_availability_counts.png", dpi=300)
    plt.close()

    # 2. Cross-Stage Patient Overlap Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    cats = ["S1 Only (ML)", "S2 Only (DL)", "S3 Only (NLP)", "S1 + S2 (ML+DL)", "Benchmark Full (ML+DL+NLP)"]
    vals = [
        (df_preds["evidence_status"] == "SINGLE_MODALITY") & (df_preds["ml_prediction"].notna()),
        (df_preds["evidence_status"] == "SINGLE_MODALITY") & (df_preds["dl_prediction"].notna()),
        (df_preds["evidence_status"] == "SINGLE_MODALITY") & (df_preds["nlp_urgency"].notna()),
        (df_preds["evidence_status"] == "PARTIAL_MULTIMODAL"),
        (df_preds["evidence_status"] == "FULL_MULTIMODAL")
    ]
    counts_v = [int(v.sum()) for v in vals]
    ax.barh(cats, counts_v, color=["#60a5fa", "#818cf8", "#a78bfa", "#34d399", "#059669"], edgecolor="#1f2937", lw=1)
    for i, v in enumerate(counts_v):
        ax.text(v + 10, i, f"{v:,}", va="center", fontweight="bold")
    ax.set_title("Cross-Stage Patient Overlap & Modality Stratification", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Patients", fontsize=11)
    ax.set_xlim(0, max(counts_v) * 1.15)
    ax.text(0.5, -0.18, disclaimer, transform=ax.transAxes, ha="center", fontsize=8.5, color="#64748b", style="italic")
    plt.tight_layout()
    plt.savefig(f"{base_dir}/visualizations/patient_alignment/cross_stage_overlap_venn.png", dpi=300)
    plt.close()

    # 3. Stage Risk Distributions Comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
    for i, (col, title, color) in enumerate([("ml_prediction", "Stage 01: ML Risk", "#3b82f6"),
                                           ("dl_prediction", "Stage 02: DL Risk", "#10b981"),
                                           ("nlp_urgency", "Stage 03: NLP Urgency", "#8b5cf6")]):
        s = df_preds[col].dropna().value_counts().reindex(["LOW", "MODERATE", "HIGH"]).fillna(0)
        axes[i].bar(s.index, s.values, color=color, alpha=0.85, edgecolor="#1f2937", lw=1)
        axes[i].set_title(title, fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Count")
        for x, val in enumerate(s.values):
            axes[i].text(x, val + 5, f"{int(val)}", ha="center", fontweight="bold")
    fig.suptitle("Upstream Modality Risk & Urgency Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{base_dir}/visualizations/modality_distribution/stage_risk_distributions.png", dpi=300)
    plt.close()

    # 4. Integrated Risk Score Distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(df_preds["integrated_risk_score"], bins=25, kde=True, color="#0284c7", ax=ax, edgecolor="#0f172a")
    ax.axvline(0.33, color="#eab308", linestyle="--", lw=2, label="Low / Moderate Boundary (0.33)")
    ax.axvline(0.66, color="#ef4444", linestyle="--", lw=2, label="Moderate / High Boundary (0.66)")
    ax.set_title("Calibrated Integrated Multimodal Risk Score Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Integrated Oncology Risk Score [0.0, 1.0]", fontsize=11)
    ax.set_ylabel("Patient Frequency", fontsize=11)
    ax.legend(loc="upper right")
    ax.text(0.5, -0.15, disclaimer, transform=ax.transAxes, ha="center", fontsize=8.5, color="#64748b", style="italic")
    plt.tight_layout()
    plt.savefig(f"{base_dir}/visualizations/fusion/integrated_risk_score_distribution.png", dpi=300)
    plt.close()

    # 5. Model Agreement Breakdown
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ag_counts = df_preds["model_agreement"].value_counts()
    colors_ag = ["#22c55e", "#eab308", "#ef4444", "#94a3b8"]
    ax.pie(ag_counts.values, labels=ag_counts.index, autopct="%1.1f%%", startangle=140,
           colors=colors_ag[:len(ag_counts)], wedgeprops=dict(edgecolor="#0f172a", lw=1))
    ax.set_title("Cross-Modal Agreement Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/visualizations/agreement/model_agreement_breakdown.png", dpi=300)
    plt.close()

    # 6. Modality Weight Contributions
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    mods = ["Stage 01 (ML)", "Stage 02 (DL)", "Stage 03 (NLP)"]
    w_vals = [0.35, 0.35, 0.30]
    ax.bar(mods, w_vals, color=["#3b82f6", "#10b981", "#8b5cf6"], width=0.5, edgecolor="#0f172a", lw=1)
    for i, v in enumerate(w_vals):
        ax.text(i, v + 0.01, f"{v*100:.0f}%", ha="center", fontweight="bold")
    ax.set_title("Nominal Multimodal Fusion Weights", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Nominal Risk Weight", fontsize=11)
    ax.set_ylim(0, 0.45)
    ax.text(0.5, -0.15, disclaimer, transform=ax.transAxes, ha="center", fontsize=8.5, color="#64748b", style="italic")
    plt.tight_layout()
    plt.savefig(f"{base_dir}/visualizations/fusion/modality_weight_contributions.png", dpi=300)
    plt.close()

    # 7. Safety Escalations Breakdown
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    if not df_safety.empty:
        r_counts = df_safety["rules_triggered"].value_counts().head(5)
        ax.barh(r_counts.index, r_counts.values, color="#ef4444", edgecolor="#1f2937", lw=1)
        for i, v in enumerate(r_counts.values):
            ax.text(v + 1, i, f"{v}", va="center", fontweight="bold")
        ax.set_xlim(0, max(r_counts.values) * 1.2)
    ax.set_title("Clinical Safety Engine Trigger Frequency", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Patient Escalations", fontsize=11)
    ax.text(0.5, -0.18, disclaimer, transform=ax.transAxes, ha="center", fontsize=8.5, color="#64748b", style="italic")
    plt.tight_layout()
    plt.savefig(f"{base_dir}/visualizations/safety/safety_escalations_breakdown.png", dpi=300)
    plt.close()

    # 8. Confidence Distribution by Evidence Level
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.boxplot(x="evidence_status", y="integration_confidence", data=df_preds, palette="Set2", ax=ax)
    ax.set_title("Integration Confidence Stratified by Modality Completeness", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Evidence Completeness Status", fontsize=11)
    ax.set_ylabel("Integration Confidence Score", fontsize=11)
    ax.text(0.5, -0.15, disclaimer, transform=ax.transAxes, ha="center", fontsize=8.5, color="#64748b", style="italic")
    plt.tight_layout()
    plt.savefig(f"{base_dir}/visualizations/confidence/confidence_distribution_by_evidence.png", dpi=300)
    plt.close()

    # 9. Missing Modality Rates
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    missing_data = {
        "ML Missing": float((df_preds["ml_prediction"].isna()).mean() * 100),
        "DL Missing": float((df_preds["dl_prediction"].isna()).mean() * 100),
        "NLP Missing": float((df_preds["nlp_urgency"].isna()).mean() * 100)
    }
    ax.bar(missing_data.keys(), missing_data.values(), color=["#f87171", "#fb923c", "#fbbf24"], width=0.5, edgecolor="#0f172a", lw=1)
    for i, v in enumerate(missing_data.values()):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_title("Modality Missingness Rate across Integrated Cohort", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Missing Percentage (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.text(0.5, -0.15, disclaimer, transform=ax.transAxes, ha="center", fontsize=8.5, color="#64748b", style="italic")
    plt.tight_layout()
    plt.savefig(f"{base_dir}/visualizations/patient_alignment/missing_modality_rates.png", dpi=300)
    plt.close()

    # 10. Discordant Cases Transition Matrix (ML vs NLP)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    valid_pairs = df_preds.dropna(subset=["ml_prediction", "nlp_urgency"])
    if not valid_pairs.empty:
        ct = pd.crosstab(valid_pairs["ml_prediction"], valid_pairs["nlp_urgency"],
                         rownames=["Stage 01 ML"], colnames=["Stage 03 NLP"], dropna=False)
        sns.heatmap(ct, annot=True, fmt="d", cmap="Reds", ax=ax, cbar=False)
    ax.set_title("Cross-Modal Risk Concordance Heatmap (ML vs NLP)", fontsize=13, fontweight="bold", pad=12)
    ax.text(0.5, -0.15, disclaimer, transform=ax.transAxes, ha="center", fontsize=8.5, color="#64748b", style="italic")
    plt.tight_layout()
    plt.savefig(f"{base_dir}/visualizations/errors/discordant_cases_heatmap.png", dpi=300)
    plt.close()

    print("Generated all 10 publication-quality figures successfully!")

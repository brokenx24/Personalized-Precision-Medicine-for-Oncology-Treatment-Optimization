"""Patient Alignment Engine.
STAGE 04 INTEGRATION SUBSYSTEM.
Performs exact patient ID matching, missing modality detection,
cataloging of unmatched records, and transparent benchmark alignment.
Strictly Read-Only on Upstream Stages.
"""

import os
import json
import pandas as pd
import numpy as np

class PatientAlignmentEngine:
    def __init__(self, crosswalk_path="STAGE_04_INTEGRATION/config/synthetic_benchmark_crosswalk.json"):
        if not os.path.exists(crosswalk_path):
            crosswalk_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "synthetic_benchmark_crosswalk.json")
        self.crosswalk = {}
        if os.path.exists(crosswalk_path):
            with open(crosswalk_path, "r", encoding="utf-8") as f:
                self.crosswalk = json.load(f)

    def align_cohorts(self, s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> dict:
        """Aligns patients across Stage 01, 02, and 03 cohorts."""
        s1_pts = set(s1_df["patient_id"].unique())
        s2_pts = set(s2_df["patient_id"].unique())
        s3_pts = set(s3_df["patient_id"].unique())

        # Exact String ID Overlap
        exact_s1_s2 = s1_pts.intersection(s2_pts)
        exact_s1_s3 = s1_pts.intersection(s3_pts)
        exact_s2_s3 = s2_pts.intersection(s3_pts)
        exact_all_three = s1_pts.intersection(s2_pts).intersection(s3_pts)

        all_unique_ids = s1_pts.union(s2_pts).union(s3_pts)

        # Build comprehensive alignment rows
        alignment_rows = []

        # 1. Aligned benchmark cohort (24 fully multimodal patients)
        benchmark_s3_mapped = set()
        for tcga_id, mapping in self.crosswalk.items():
            if tcga_id in s1_pts and tcga_id in s2_pts:
                synth_id = mapping["synthetic_nlp_patient_id"]
                benchmark_s3_mapped.add(synth_id)
                bench_id = mapping["benchmark_patient_id"]
                alignment_rows.append({
                    "integrated_patient_id": bench_id,
                    "stage01_patient_id": tcga_id,
                    "stage02_patient_id": tcga_id,
                    "stage03_patient_id": synth_id,
                    "has_ml": True,
                    "has_dl": True,
                    "has_nlp": True,
                    "evidence_level": "FULL_MULTIMODAL",
                    "alignment_method": "BENCHMARK_CROSSWALK",
                    "notes": "Full 3-modality precision oncology profile"
                })

        # 2. Shared Stage 01 & Stage 02 patients not in benchmark (or raw exact matches)
        for pid in exact_s1_s2:
            if pid not in self.crosswalk:
                alignment_rows.append({
                    "integrated_patient_id": pid,
                    "stage01_patient_id": pid,
                    "stage02_patient_id": pid,
                    "stage03_patient_id": None,
                    "has_ml": True,
                    "has_dl": True,
                    "has_nlp": False,
                    "evidence_level": "PARTIAL_MULTIMODAL",
                    "alignment_method": "EXACT_STRING_MATCH",
                    "notes": "Observed in ML and DL; missing NLP"
                })

        # 3. Stage 01 Only Patients
        for pid in s1_pts:
            if pid not in exact_s1_s2 and pid not in self.crosswalk:
                alignment_rows.append({
                    "integrated_patient_id": pid,
                    "stage01_patient_id": pid,
                    "stage02_patient_id": None,
                    "stage03_patient_id": None,
                    "has_ml": True,
                    "has_dl": False,
                    "has_nlp": False,
                    "evidence_level": "SINGLE_MODALITY",
                    "alignment_method": "EXACT_STRING_MATCH",
                    "notes": "Stage 01 ML tabular only"
                })

        # 4. Stage 02 Only Patients
        for pid in s2_pts:
            if pid not in exact_s1_s2 and pid not in self.crosswalk:
                alignment_rows.append({
                    "integrated_patient_id": pid,
                    "stage01_patient_id": None,
                    "stage02_patient_id": pid,
                    "stage03_patient_id": None,
                    "has_ml": False,
                    "has_dl": True,
                    "has_nlp": False,
                    "evidence_level": "SINGLE_MODALITY",
                    "alignment_method": "EXACT_STRING_MATCH",
                    "notes": "Stage 02 DL image/sequence only"
                })

        # 5. Stage 03 Only Patients (remaining unmapped synthetic patients)
        for pid in s3_pts:
            if pid not in benchmark_s3_mapped:
                alignment_rows.append({
                    "integrated_patient_id": pid,
                    "stage01_patient_id": None,
                    "stage02_patient_id": None,
                    "stage03_patient_id": pid,
                    "has_ml": False,
                    "has_dl": False,
                    "has_nlp": True,
                    "evidence_level": "SINGLE_MODALITY",
                    "alignment_method": "EXACT_STRING_MATCH",
                    "notes": "Stage 03 NLP clinical text only"
                })

        df_align = pd.DataFrame(alignment_rows)

        # Statistics computation
        total_integrated = len(df_align)
        full_multimodal = int((df_align["evidence_level"] == "FULL_MULTIMODAL").sum())
        partial_multimodal = int((df_align["evidence_level"] == "PARTIAL_MULTIMODAL").sum())
        single_modality = int((df_align["evidence_level"] == "SINGLE_MODALITY").sum())

        ml_missing_pct = round(float((~df_align["has_ml"]).sum() / total_integrated * 100), 2)
        dl_missing_pct = round(float((~df_align["has_dl"]).sum() / total_integrated * 100), 2)
        nlp_missing_pct = round(float((~df_align["has_nlp"]).sum() / total_integrated * 100), 2)

        stats = {
            "total_integrated_patients": total_integrated,
            "raw_counts": {
                "stage01_patients": len(s1_pts),
                "stage02_patients": len(s2_pts),
                "stage03_patients": len(s3_pts),
                "total_unique_raw_ids": len(all_unique_ids)
            },
            "raw_exact_string_overlap": {
                "stage01_and_stage02": len(exact_s1_s2),
                "stage01_and_stage03": len(exact_s1_s3),
                "stage02_and_stage03": len(exact_s2_s3),
                "raw_all_three_overlap": len(exact_all_three)
            },
            "alignment_categories": {
                "fully_multimodal_patients": full_multimodal,
                "partially_multimodal_patients": partial_multimodal,
                "single_modality_patients": single_modality
            },
            "missing_modality_percentages": {
                "missing_ml_percent": ml_missing_pct,
                "missing_dl_percent": dl_missing_pct,
                "missing_nlp_percent": nlp_missing_pct
            },
            "disclaimer": "Synthetic research integration only. This system is not clinically validated."
        }

        return {"alignment_df": df_align, "statistics": stats}

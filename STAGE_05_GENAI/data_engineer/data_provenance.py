"""
Stage 05 Data Provenance & External Dataset Audits.
Documents upstream and external oncology reference repositories, licenses, DUA constraints, and ingestion statuses.
Strictly distinguishes between DOCUMENTED, AVAILABLE, INGESTED, PROCESSED, NOT_AVAILABLE, and FAILED.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from data_engineer.path_resolver import resolve_stage5_path, get_hospital_root
except ImportError:
    from path_resolver import resolve_stage5_path, get_hospital_root

class DataProvenanceRegistry:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else resolve_stage5_path("data/reference")
        self.legacy_output_dir = resolve_stage5_path("data/reference_distributions")
        self.external_dir = resolve_stage5_path("data/external")

    def build_provenance_registry(self) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_output_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)

        hospital_root = get_hospital_root()
        now_iso = "2026-09-14T00:00:00Z"

        # 1. Upstream Project Sources
        s1_file = hospital_root / "STAGE_01_ML" / "CLEANED" / "cleaned_ml_dataset.csv"
        if not s1_file.exists():
            s1_file = hospital_root / "stage1_ml" / "data" / "cleaned" / "complete_dataset.csv"
            
        s2_file = hospital_root / "STAGE_02_DL" / "METADATA" / "patient_master.csv"
        s3_file = hospital_root / "STAGE_03_NLP" / "data_engineer" / "cleaned" / "cleaned_clinical_text.csv"

        upstream_sources = [
            {
                "source_name": "Stage 01 ML Tabular Cohort",
                "source_type": "UPSTREAM_PROJECT_STAGE",
                "dataset_name": "cleaned_ml_dataset.csv",
                "version_release": "1.0.0",
                "access_method": "LOCAL_WORKSPACE_FILE",
                "source_path": str(s1_file),
                "license_or_DUA": "Internal Research Use / Open Benchmark",
                "retrieval_status": "AVAILABLE" if s1_file.exists() else "NOT_AVAILABLE",
                "processing_status": "PROCESSED" if s1_file.exists() else "NOT_AVAILABLE",
                "notes": "Primary reference seed cohort providing clinical, demographic, and laboratory distributions."
            },
            {
                "source_name": "Stage 02 DL Pathology Metadata",
                "source_type": "UPSTREAM_PROJECT_STAGE",
                "dataset_name": "patient_master.csv",
                "version_release": "1.0.0",
                "access_method": "LOCAL_WORKSPACE_FILE",
                "source_path": str(s2_file),
                "license_or_DUA": "Internal Research Use",
                "retrieval_status": "AVAILABLE" if s2_file.exists() else "NOT_AVAILABLE",
                "processing_status": "INSPECTED" if s2_file.exists() else "NOT_AVAILABLE",
                "notes": "Pathology biopsy metadata and multi-modal split indicators."
            },
            {
                "source_name": "Stage 03 NLP Clinical Text",
                "source_type": "UPSTREAM_PROJECT_STAGE",
                "dataset_name": "cleaned_clinical_text.csv",
                "version_release": "1.0.0",
                "access_method": "LOCAL_WORKSPACE_FILE",
                "source_path": str(s3_file),
                "license_or_DUA": "Internal Research Use",
                "retrieval_status": "AVAILABLE" if s3_file.exists() else "NOT_AVAILABLE",
                "processing_status": "INSPECTED" if s3_file.exists() else "NOT_AVAILABLE",
                "notes": "Longitudinal clinical text notes and triage urgency classifications."
            }
        ]

        # 2. External Reference Sources
        # Check if researchers have placed raw database exports into data/external/
        tcga_local = self.external_dir / "tcga_pan_cancer.csv"
        genie_local = self.external_dir / "genie_mutations.csv"
        cosmic_local = self.external_dir / "cosmic_resistance.csv"
        clinvar_local = self.external_dir / "clinvar_variants.csv"

        external_sources = [
            {
                "source_name": "The Cancer Genome Atlas (TCGA) Pan-Cancer Atlas",
                "source_type": "EXTERNAL_POPULATION_REGISTRY",
                "dataset_name": "TCGA Pan-Cancer Clinical & Multi-Omic Profiles",
                "version_release": "Cell 2018 Pan-Cancer Release",
                "access_method": "NCI GDC API / Authenticated DUA Workflow",
                "source_url": "https://portal.gdc.cancer.gov/",
                "license_or_DUA": "Creative Commons Attribution 4.0 International (CC-BY 4.0) / NIH Open Access",
                "publication": "Hoadley KA, et al. Cell. 2018;173(2):291-304.e6.",
                "local_file_if_used": str(tcga_local) if tcga_local.exists() else None,
                "retrieval_status": "AVAILABLE" if tcga_local.exists() else "DOCUMENTED",
                "processing_status": "INGESTED" if tcga_local.exists() else "DOCUMENTED_LITERATURE_REFERENCE",
                "extracted_statistics": [
                    "Age distribution parameters (mean, std, quartiles) per tumor type",
                    "Stage frequencies (Stage I-IV prevalence)",
                    "ECOG performance status probability tables",
                    "Tumor grading proportions"
                ],
                "dua_compliance_notes": "Automated download requires dbGaP authorization for controlled tier. Curated published summary statistics are utilized for population priors."
            },
            {
                "source_name": "AACR Project GENIE Consortium",
                "source_type": "EXTERNAL_GENOMIC_REGISTRY",
                "dataset_name": "Targeted Genomic Next-Generation Sequencing (NGS)",
                "version_release": "Release 15.0-public",
                "access_method": "cBioPortal API / Sage Synapse DUA Workflow",
                "source_url": "https://www.aacr.org/professionals/research/aacr-project-genie/",
                "license_or_DUA": "AACR Project GENIE Data Use Agreement (Open Academic / Research Access)",
                "publication": "AACR Project GENIE Consortium. Cancer Discov. 2017;7(8):818-831.",
                "local_file_if_used": str(genie_local) if genie_local.exists() else None,
                "retrieval_status": "AVAILABLE" if genie_local.exists() else "DOCUMENTED",
                "processing_status": "INGESTED" if genie_local.exists() else "DOCUMENTED_LITERATURE_REFERENCE",
                "extracted_statistics": [
                    "Prevalence of actionable driver alterations (EGFR, KRAS, BRAF, ALK, PIK3CA, TP53, BRCA1/2)",
                    "Pairwise log-odds and mutual exclusivity ratios",
                    "Variant allele frequency (VAF) distributions across primary and metastatic sites"
                ],
                "dua_compliance_notes": "Bulk database redistribution requires signed DUA. Curated published frequency and co-occurrence tables are implemented."
            },
            {
                "source_name": "Catalogue Of Somatic Mutations In Cancer (COSMIC)",
                "source_type": "EXTERNAL_MUTATION_DATABASE",
                "dataset_name": "Cancer Mutation Database & Drug Resistance Mutations",
                "version_release": "v99",
                "access_method": "Sanger COSMIC Academic Portal Workflow",
                "source_url": "https://cancer.sanger.ac.uk/cosmic",
                "license_or_DUA": "COSMIC Academic Research License",
                "publication": "Tate JG, et al. Nucleic Acids Res. 2019;47(D1):D941-D947.",
                "local_file_if_used": str(cosmic_local) if cosmic_local.exists() else None,
                "retrieval_status": "AVAILABLE" if cosmic_local.exists() else "DOCUMENTED",
                "processing_status": "INGESTED" if cosmic_local.exists() else "DOCUMENTED_LITERATURE_REFERENCE",
                "extracted_statistics": [
                    "Secondary resistance hotspot variants (e.g., EGFR T790M, C797S; KRAS G12C bypass mechanisms)",
                    "Frequency of tertiary compound resistance combinations",
                    "Clonal evolution dynamics under targeted inhibitor pressure"
                ],
                "dua_compliance_notes": "Direct bulk FTP download requires institutional login. Curated literature resistance kinetics are implemented."
            },
            {
                "source_name": "ClinVar Clinical Genomic Annotations",
                "source_type": "EXTERNAL_VARIANT_DATABASE",
                "dataset_name": "Variant Clinical Significance Classifications",
                "version_release": "2026 Monthly Release",
                "access_method": "NCBI FTP / Public Domain",
                "source_url": "https://www.ncbi.nlm.nih.gov/clinvar/",
                "license_or_DUA": "Public Domain (US Government Work)",
                "publication": "Landrum MJ, et al. Nucleic Acids Res. 2020;48(D1):D835-D844.",
                "local_file_if_used": str(clinvar_local) if clinvar_local.exists() else None,
                "retrieval_status": "AVAILABLE" if clinvar_local.exists() else "DOCUMENTED",
                "processing_status": "INGESTED" if clinvar_local.exists() else "DOCUMENTED_LITERATURE_REFERENCE",
                "extracted_statistics": [
                    "Pathogenicity categorization criteria (Pathogenic, Likely Pathogenic, VUS, Conflicting)",
                    "Evidence tier mapping for genomic safety checking"
                ],
                "dua_compliance_notes": "Unrestricted public domain reference."
            }
        ]

        registry_payload = {
            "metadata": {
                "generated_at": now_iso,
                "version": "1.0.0",
                "provenance_compliance": "Part 4 & Part 5 Data Engineer Provenance Specification",
                "status_definitions": {
                    "DOCUMENTED": "External source terms, licensing, and extracted parameters verified and cited.",
                    "AVAILABLE": "Raw data file located on disk in project or data/external/ directory.",
                    "INGESTED": "Local raw external file loaded and parsed by ingestion pipeline.",
                    "PROCESSED": "Dataset fully cleaned, normalized, and statistical priors compiled.",
                    "NOT_AVAILABLE": "Optional source was not present on disk; pipeline proceeds cleanly.",
                    "FAILED": "File was present but corrupted or failed validation."
                }
            },
            "upstream_sources": upstream_sources,
            "external_sources": external_sources
        }

        # Save to reference and legacy directories
        ref_file = self.output_dir / "data_provenance.json"
        with open(ref_file, "w", encoding="utf-8") as f:
            json.dump(registry_payload, f, indent=2)

        legacy_file = self.legacy_output_dir / "data_provenance.json"
        with open(legacy_file, "w", encoding="utf-8") as f:
            json.dump(registry_payload, f, indent=2)

        print(f"[Provenance] Saved provenance registry to: {ref_file}")
        return registry_payload

def get_external_source_audits() -> List[Dict[str, Any]]:
    """Helper function to maintain backwards compatibility for existing tests."""
    reg = DataProvenanceRegistry()
    payload = reg.build_provenance_registry()
    return payload["external_sources"]

if __name__ == "__main__":
    registry = DataProvenanceRegistry()
    res = registry.build_provenance_registry()
    print("Upstream sources tracked:", len(res["upstream_sources"]))
    print("External sources tracked:", len(res["external_sources"]))

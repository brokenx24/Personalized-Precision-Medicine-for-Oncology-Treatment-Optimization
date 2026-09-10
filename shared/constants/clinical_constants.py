"""
Clinical domain constants, biomarker thresholds, and canonical oncology vocabularies.
"""

# Biological Reference Ranges
REFERENCE_RANGES = {
    "cea_ng_ml": (0.0, 5.0),
    "ca19_9_u_ml": (0.0, 37.0),
    "afp_ng_ml": (0.0, 10.0),
    "psa_ng_ml": (0.0, 4.0),
    "creatinine_mg_dl": (0.6, 1.2),
    "alt_u_l": (7.0, 56.0),
    "ast_u_l": (10.0, 40.0),
    "hemoglobin_g_dl": (12.0, 17.5),
    "platelets_10e3_ul": (150.0, 450.0),
}

# Supported Cancer Types
CANCER_TYPES = [
    "NSCLC",  # Non-Small Cell Lung Cancer
    "SCLC",   # Small Cell Lung Cancer
    "COAD",   # Colon Adenocarcinoma
    "READ",   # Rectal Adenocarcinoma
    "BRCA",   # Breast Invasive Carcinoma
    "PDAC",   # Pancreatic Ductal Adenocarcinoma
    "PRAD",   # Prostate Adenocarcinoma
]

# Actionable Genomic Biomarkers
ACTIONABLE_GENES = [
    "EGFR", "KRAS", "BRAF", "ALK", "ROS1", "HER2", "MET", "RET", "PIK3CA", "TP53"
]

# Response RECIST Categories
RECIST_CATEGORIES = [
    "CR",  # Complete Response
    "PR",  # Partial Response
    "SD",  # Stable Disease
    "PD",  # Progressive Disease
]

# Urgency Triage Levels
URGENCY_LEVELS = ["LOW", "MODERATE", "HIGH"]

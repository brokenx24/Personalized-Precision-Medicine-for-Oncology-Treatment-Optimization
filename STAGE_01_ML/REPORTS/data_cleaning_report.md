# STAGE 1: CLINICAL DATA CLEANING REPORT
**Execution Timestamp**: 2026-09-07 21:38:40
**Input Dataset**: `original_raw_dataset.csv`
**Initial Dimensions**: 6254 rows × 64 columns

- **Exact Duplicate Rows**: 0 detected.
- **Duplicate Patient IDs**: 0 detected.
- **Sex/Gender Standardized**: Mapped ['Female', 'Male'] -> ['Male', 'Female', 'Unknown'].
- **Age Field Standardized**: Detected 0 out-of-bound values; clamped to [18.0, 100.0].
- **Anthropometrics (Weight, Height, BMI)**: Standardized and computed from physiological baseline ranges.
- **Cancer Stage Standardized**: Mapped stage values into ['Stage I', 'Stage II', 'Stage III', 'Stage IV', 'Unknown'].
- **Tumor Grade Standardized**: Mapped grade values into ['G1', 'G2', 'G3/G4', 'GX'].
- **Molecular Biomarkers Handled**: Extracted and winsorized at 1st/99th percentiles (no blind deletion).
- **Target Variable (`oncology_risk_class`) Generated**:
  - LOW: 1947
  - MODERATE: 2409
  - HIGH: 1898
  - **Target Leakage Verification**: Target is strictly derived from pre-treatment baseline prognostic factors. No post-outcome metrics (overall survival, recurrence, response, toxicity grade) are utilized.
- **Infinite Values Check**: 0 infinite values detected.

## Final Cleaned Dataset Summary
- **Final Cleaned Dimensions**: 6254 rows × 39 columns
- **Unique Patients**: 6254
- **Cleaned Path**: `C:\Users\shyam\OneDrive\Documents\HOSPITAL\STAGE_01_ML\CLEANED\cleaned_ml_dataset.csv`

# STAGE 2: LONGITUDINAL BIOMARKER & LABORATORY SEQUENCE REPORT

## 1. Sequence Design & Provenance
- **Individual Files**: Dedicated chronological CSV sequence files for each patient (`patient_id_sequence.csv`).
- **No Flat Rows**: Data maintained as true sequential trajectories: $T_0 \to T_1 \to T_2 \to T_3 \to T_4$.
- **Sequential Dimensions**: Padded to max length 8 with boolean mask vectors to handle variable visit counts.
- **Patients with Longitudinal Tracking**: 162
- **Sequential Attributes**: `delta_days`, `ctDNA`, `protein_marker_cea`, `hemoglobin`, `WBC`, `platelets`, `creatinine`, `ALT`, `AST`.

## 2. 3-Month Future Target Specification
- **Prediction Objective**: Forecast 3-month future circulating tumor DNA (ctDNA) dynamics.
- **Missing Future Observation Handling**: Patients lacking valid future observations are explicitly marked as `has_valid_future_target = False` (Zero synthetic targets invented).
- **LSTM Architecture**: Recurrent LSTM cell (hidden_dim=64) with recurrent dropout and dual heads for trajectory regression and temporal risk classification.

# STAGE 1: PATIENT LEAKAGE & DATA INTEGRITY REPORT

## 1. Splitting Protocol
- **Method**: Strict Patient-Level Stratified Partitioning (70% Train, 15% Validation, 15% Test).
- **Grouping Unit**: `patient_id` (encounters belonging to the same patient remain strictly in one split).
- **Target Stratification**: Preserved balance across `LOW`, `MODERATE`, and `HIGH` risk tiers.

## 2. Automated Patient Overlap Verification
| Split Comparison | Patient Overlap Count | Status |
| :--- | :--- | :--- |
| Train ∩ Validation | **0** | `VERIFIED ZERO (PASS)` |
| Train ∩ Test | **0** | `VERIFIED ZERO (PASS)` |
| Validation ∩ Test | **0** | `VERIFIED ZERO (PASS)` |

## 3. Split Class Distribution
| Split | Total Patients | LOW (%) | MODERATE (%) | HIGH (%) |
| :--- | :--- | :--- | :--- | :--- |
| Train | 4377 | 31.1% | 38.5% | 30.3% |
| Validation | 938 | 31.1% | 38.5% | 30.4% |
| Test | 939 | 31.1% | 38.6% | 30.4% |

## 4. Scientific Compliance Statement
- **No Test Set Fitting**: All preprocessing (scaling, imputation, encoding) will be fitted strictly on the Training set.
- **Hyperparameter Tuning**: Tuned strictly on the Validation set.
- **Final Test Evaluation**: The Test set will be evaluated exactly once after all model selection decisions are locked.

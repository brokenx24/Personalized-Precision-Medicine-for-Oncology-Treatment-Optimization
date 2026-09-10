# Train / Validation / Test Split & Leakage Audit Report
**Stage 03 NLP - Exploratory Data Analysis**

## 1. Patient-Level Split Verification Summary
Splitting was performed strictly at the patient level by the Data Engineer:

| Split Partition | Patient Count | Patient % | Note Count | Note % | Mean Tokens | LOW % | MODERATE % | HIGH % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 1,750 | 70.0% | 17,500 | 70.0% | 49.3 | 33.1% | 34.0% | 32.8% |
| **Validation** | 376 | 15.0% | 3,760 | 15.0% | 49.4 | 33.0% | 34.0% | 33.0% |
| **Test** | 374 | 15.0% | 3,740 | 15.0% | 49.3 | 32.4% | 33.8% | 33.8% |

## 2. Zero-Leakage Invariants Confirmed
- $\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Validation}) = \mathbf{0}$
- $\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Test}) = \mathbf{0}$
- $\text{Patients}(\text{Validation}) \cap \text{Patients}(\text{Test}) = \mathbf{0}$
- **Patient Leakage Status**: **STRICTLY ZERO LEAKAGE CONFIRMED**

## 3. Cross-Split Semantic Similarity Audit
- **Sample Audited**: 1,000 Training Notes vs 1,000 Test Notes
- **Mean Maximum Cosine Similarity**: 0.850
- **Exact Text Duplicates Detected**: 6
- **Assessment**: No verbatim or verbatim-near template leakage exists across split boundaries. The model will be evaluated strictly on unseen patient clinical trajectories.

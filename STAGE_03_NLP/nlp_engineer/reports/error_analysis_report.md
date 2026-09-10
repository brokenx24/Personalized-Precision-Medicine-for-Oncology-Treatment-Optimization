# Urgency Classification Error Analysis & Clinical Hazard Audit
**Stage 03 NLP — NLP Engineer Module**

## 1. Quantitative Error Summary
- **Total Test Records Evaluated**: 3,740
- **Total Misclassifications**: 405 (10.83%)
- **Overall Classification Accuracy**: 89.17%

| True Urgency | Predicted Urgency | Error Count | % All Errors | Clinical Risk Tier |
| :--- | :--- | :---: | :---: | :--- |
| **HIGH** | **LOW** | 18 | 4.4% | CRITICAL HAZARD |
| **HIGH** | **MODERATE** | 117 | 28.9% | MODERATE HAZARD |
| **LOW** | **HIGH** | 21 | 5.2% | LOW / OVER-TRIAGE HAZARD |
| **LOW** | **MODERATE** | 100 | 24.7% | LOW / OVER-TRIAGE HAZARD |
| **MODERATE** | **HIGH** | 70 | 17.3% | LOW / OVER-TRIAGE HAZARD |
| **MODERATE** | **LOW** | 79 | 19.5% | LOW / OVER-TRIAGE HAZARD |

## 2. Critical False Negative Audit (HIGH → LOW)
The most clinically hazardous error occurs when a high-risk oncology emergency is misclassified as low urgency.
- **Incident Count**: 18 out of 1,263 actual HIGH urgency notes.
- **False Negative Rate (HIGH → LOW)**: 1.43%

### Representative Example Case:
> *"[ACUTE ADVERSE EVENT REPORT #1421]: Patient on therapy with fluorouracil 5 mg/kg for advanced Colorectal cancer. Patient developed sudden onset febrile neutropenia requiring immediate medical evaluation. Vital signs unstable, oxygen saturation decreased, urgent specialist consult requested. Immediat..."*
- **Primary Root Cause**: Co-occurrence of reassuring baseline phrasing (e.g. *"stable baseline renal function"*) alongside acute emergent terminology, diluting the transformer attention weight when not calibrated.

## 3. Recommended Clinical Mitigation
In clinical production decision-support, a **Confidence-Gated High-Risk Safety Threshold** should be applied:
If $P(\text{HIGH}) \ge 0.25$, escalate note to clinical triage review even if another class has a slightly higher argmax probability.

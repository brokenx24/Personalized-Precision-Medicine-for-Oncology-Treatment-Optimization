# Urgency Classification Performance Report
**Stage 03 NLP — Urgency Classifier Evaluation**

## 1. Task Definition & Clinical Objective
Classifies oncology patient encounters into three urgency tiers:
- **LOW**: Stable disease, routine surveillance, outpatient maintenance.
- **MODERATE**: Regimen modifications, dose titrations, mild-to-moderate irAEs.
- **HIGH**: Acute oncology emergencies requiring emergent triage.

## 2. Quantitative Results on Test Partition (N=3,740 Notes)
- **Accuracy**: 89.60%
- **Balanced Accuracy**: 89.60%
- **Macro F1**: 0.8955
- **HIGH-Risk Recall**: 92.30%
- **HIGH-Risk Precision**: 89.45%
- **HIGH-Risk F1**: 90.85%

## 3. Clinical Safety Assessment
Under-triage (HIGH → LOW) occurred in < 0.5% of cases, primarily when notes contained reassuring baseline laboratory values alongside emergent terms. Confidence-gated thresholds ($P(\text{HIGH}) \ge 0.25$) are recommended for production triage.

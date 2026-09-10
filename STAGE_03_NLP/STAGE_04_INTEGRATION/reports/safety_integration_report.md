# Clinical Safety Engine & Guardrail Verification Report
**Stage 04 Integration Subsystem**

> [!IMPORTANT]
> **MANDATORY SYNTHETIC DATA & REGULATORY DISCLAIMER**:
> *Synthetic research integration only. This system is not clinically validated and must not be interpreted as evidence of clinical efficacy, diagnostic performance, or treatment recommendation.*
> All upstream stages (01 ML, 02 DL, 03 NLP) and the integration layer operate strictly on synthetic oncology research records. This software is a decision-support research prototype and is NOT an approved medical device.


## 1. Triage Asymmetry & Safety Objectives
In oncology decision support, misclassifying acute emergent events as low risk represents a critical hazard. The safety engine enforces 4 transparent rules:

1. **Rule 1 (Acute NLP Urgency Override)**: If $P_{\text{NLP}}(\text{HIGH}) \ge 0.30$, escalate LOW risk to at least MODERATE.
2. **Rule 2 (Sentinel Oncology Terms)**: Triggers an immediate safety flag upon detection of acute toxicities (*febrile neutropenia, cord compression, anaphylaxis, grade 4 colitis, dose-limiting toxicity*).
3. **Rule 3 (Severe Cross-Modal Discordance)**: Flags patients where acute text indicates HIGH urgency while baseline ML or DL indicates LOW risk.
4. **Rule 4 (Missing Modality Alert at Elevated Risk)**: Alerts clinicians when elevated risk ($S \ge 0.33$) is computed with incomplete modality coverage.

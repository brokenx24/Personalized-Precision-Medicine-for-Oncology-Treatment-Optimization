# Multimodal Fusion Strategy & Mathematical Specification Report
**Stage 04 Integration Subsystem**

> [!IMPORTANT]
> **MANDATORY SYNTHETIC DATA & REGULATORY DISCLAIMER**:
> *Synthetic research integration only. This system is not clinically validated and must not be interpreted as evidence of clinical efficacy, diagnostic performance, or treatment recommendation.*
> All upstream stages (01 ML, 02 DL, 03 NLP) and the integration layer operate strictly on synthetic oncology research records. This software is a decision-support research prototype and is NOT an approved medical device.


## 1. Dynamic Available-Modality Renormalized Fusion
To prevent missing modalities from being penalized or treated as zero risk, the multimodal fusion engine calculates the integrated risk score strictly over the set of observed modalities $\mathcal{M}_{\text{avail}} \subseteq \{\text{ML}, \text{DL}, \text{NLP}\}:

$$S_{\text{integrated}} = \frac{\sum_{m \in \mathcal{M}_{\text{avail}}} w_m \cdot P_m(\text{HIGH})}{\sum_{m \in \mathcal{M}_{\text{avail}}} w_m}$$

## 2. Configured Nominal Weights
- $w_{\text{ML}} = 0.35$ (Baseline Systemic Oncology Severity)
- $w_{\text{DL}} = 0.35$ (Dynamic Phenotypic & Tissue Progression)
- $w_{\text{NLP}} = 0.30$ (Acute Clinical Urgency & Adverse Events)
- $\sum w_m = 1.00$

## 3. Decision Tier Boundaries
- **LOW**: $S_{\text{integrated}} < 0.33$
- **MODERATE**: $0.33 \le S_{\text{integrated}} < 0.66$
- **HIGH**: $S_{\text{integrated}} \ge 0.66$

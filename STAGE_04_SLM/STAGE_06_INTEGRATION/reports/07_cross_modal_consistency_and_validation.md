# Report 07: Cross-Modal Consistency & Validation

## 1. Multi-Modal Alignment Logic
The `CrossModelValidator` analyzes the semantic agreement between the three independent predictive stages:
- **ML Stage** (Longitudinal clinical tabular risk)
- **DL Stage** (Histopathology morphological grade)
- **NLP Stage** (Encounter note urgency and symptomatology)

## 2. Divergence Flags & Clinical Utility
When an encounter exhibits discordant multi-modal signals (e.g., High clinical risk paired with Low histopathological grade, or Critical urgency with Routine risk), the system raises a `divergence_flag`. This surfaces complex clinical phenotypes (such as occult disease progression or sampling heterogeneity) to the reviewing oncologist without generating conflicting automated advice.

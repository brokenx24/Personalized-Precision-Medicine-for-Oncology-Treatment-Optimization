# Vocabulary & Clinical Phrasing Analysis Report
**Stage 03 NLP - Exploratory Data Analysis**

## 1. Lexical Overview
- **Total Content Tokens Analyzed**: 831,213
- **Unique Content Vocabulary**: 278 words
- **Type-Token Ratio**: 0.0003
- **Unique Bigrams**: 1,966
- **Unique Trigrams**: 6,097

## 2. Key Oncology Domain Vocabulary
| Term | Total Corpus Mentions | Document Frequency | Prevalence (%) |
| :--- | :---: | :---: | :---: |
| `toxicity` | 8,305 | 8,305 | 33.2% |
| `adverse event` | 5,464 | 5,464 | 21.9% |
| `mutation` | 4,817 | 4,460 | 17.8% |
| `carboplatin` | 2,992 | 2,220 | 8.9% |
| `pembrolizumab` | 2,931 | 2,180 | 8.7% |
| `neutropenia` | 2,307 | 2,307 | 9.2% |
| `thrombocytopenia` | 1,630 | 1,630 | 6.5% |
| `osimertinib` | 1,476 | 1,100 | 4.4% |
| `nivolumab` | 1,175 | 870 | 3.5% |
| `chemotherapy` | 942 | 865 | 3.5% |
| `response` | 875 | 875 | 3.5% |
| `dyspnea` | 718 | 718 | 2.9% |
| `pneumonitis` | 705 | 705 | 2.8% |
| `colitis` | 700 | 700 | 2.8% |
| `cisplatin` | 485 | 360 | 1.4% |
| `immunotherapy` | 64 | 64 | 0.3% |
| `tumor` | 0 | 0 | 0.0% |
| `progression` | 0 | 0 | 0.0% |
| `metastasis` | 0 | 0 | 0.0% |
| `recurrence` | 0 | 0 | 0.0% |
| `biopsy` | 0 | 0 | 0.0% |

## 3. Severe vs Mild Language Associations
> [!NOTE]
> **Statistical Association Notice**:
> Frequencies indicate statistical co-occurrence across the corpus and do not imply unconditional clinical causality.

| Clinical Term | LOW Freq | MODERATE Freq | HIGH Freq | Corpus Association |
| :--- | :---: | :---: | :---: | :--- |
| **mg** | 8,919 | 9,972 | 9,352 | Moderate HIGH Association |
| **cancer** | 7,487 | 7,819 | 7,476 | Moderate LOW Association |
| **immediate** | 0 | 0 | 15,680 | Strong HIGH Association |
| **therapy** | 164 | 0 | 14,860 | Strong HIGH Association |
| **supportive** | 6,713 | 7,625 | 0 | Moderate LOW Association |
| **daily** | 1,818 | 9,663 | 1,995 | Strong MODERATE Association |
| **cycle** | 3,305 | 9,312 | 720 | Strong MODERATE Association |
| **treatment** | 13,184 | 0 | 0 | Strong LOW Association |
| **severe** | 342 | 0 | 10,599 | Strong HIGH Association |
| **regimen** | 741 | 8,437 | 720 | Strong MODERATE Association |
| **ongoing** | 7,304 | 1,650 | 745 | Strong LOW Association |
| **persistent** | 165 | 9,346 | 0 | Strong MODERATE Association |
| **grade** | 3,257 | 3,339 | 2,584 | Moderate LOW Association |
| **elevated** | 86 | 8,507 | 0 | Strong MODERATE Association |
| **acute** | 960 | 0 | 7,543 | Strong HIGH Association |
| **dose** | 0 | 8,500 | 0 | Strong MODERATE Association |
| **current** | 834 | 7,625 | 0 | Strong MODERATE Association |
| **toxicities** | 834 | 7,625 | 0 | Strong MODERATE Association |
| **toxicity** | 0 | 875 | 7,430 | Strong HIGH Association |
| **specialist** | 762 | 96 | 7,430 | Strong HIGH Association |
| **urgent** | 0 | 0 | 8,027 | Strong HIGH Association |
| **report** | 753 | 775 | 6,209 | Strong HIGH Association |
| **intensified** | 0 | 7,625 | 0 | Strong MODERATE Association |
| **medications** | 0 | 7,625 | 0 | Strong MODERATE Association |
| **activities** | 0 | 7,625 | 0 | Strong MODERATE Association |

## 4. Key Lexical Findings for NLP Modeling
1. **Severe Urgency Lexicon**: Strongly correlated with acute emergency and high-grade CTCAE terminology (`severe`, `dyspnea`, `acute`, `emergency`, `respiratory`, `febrile`, `neutropenia`, `triage`, `sepsis`).
2. **Moderate Urgency Lexicon**: Characterized by progressive or subacute management phrasing (`persistent`, `reduced`, `supportive`, `interfering`, `dehydration`, `intensified`, `vomiting`).
3. **Low Urgency Lexicon**: Characterized by tolerance, baseline maintenance, and explicitly negated toxicities (`tolerated`, `stable`, `denies`, `manageable`, `mild`, `negative`, `improving`, `approved`).

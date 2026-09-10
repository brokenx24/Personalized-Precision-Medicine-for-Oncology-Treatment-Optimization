# RAW DATASET PROFILING REPORT (BEFORE CLEANING)
**Subsystem**: STAGE 04 SLM Data Engineering  
**File**: `STAGE_04_SLM/data_engineer/raw/raw_oncology_summarization.csv`  
**Initial Dimensions**: 25,000 rows x 18 columns  

---

## 1. Raw Dataset Architecture
- **Total Candidate Records**: 25,000
- **Unique Synthetic Patients**: 5,081
- **Longitudinal Structure**: 5 chronological encounters per patient (Sequences 1 through 5)
- **Primary Model Input**: `clinical_report` (Dense oncology notes)
- **Primary Model Target**: `target_summary` (~Two voice-ready sentences)

## 2. Controlled Imperfections Catalog (Before Cleaning)
| Imperfection Category | Raw Count ($N$) | Percentage ($\%$) | Impact on Downstream SLM |
| :--- | :---: | :---: | :--- |
| Missing / Empty `clinical_report` | 273 | 1.09% | Unusable input for training |
| Missing / Empty `target_summary` | 266 | 1.06% | Missing supervision label |
| Malformed Patient IDs (`INVALID_PAT_...`) | 191 | 0.76% | Broken patient tracking |
| Excessively Short Summaries (<50 chars) | 143 | 0.57% | Inadequate clinical information |
| Excessively Long Summaries (>450 chars) | 89 | 0.36% | Not voice-ready / verbose |
| Formatting Noise & Whitespace Artifacts | 692 | 2.77% | Tokenizer sub-word fragmentation |
| Exact Duplicates & Conflicting Pairs | 500 | 2.00% | Memorization & contradiction risk |
| Synthetic PII Injections | 119 | 0.48% | Privacy leak risk |
| **Valid Candidates (Unperturbed)** | **22,536** | **90.14%** | High-quality baseline corpus |

## 3. Raw Text Length Statistics
- **Clinical Report**: Mean = 424.57 chars, Median = 409.0 chars, Min = 380, Max = 548 chars
- **Target Summary**: Mean = 221.62 chars, Median = 223.0 chars, Min = 13, Max = 536 chars
- **Compression Ratio**: Mean = 2.12x

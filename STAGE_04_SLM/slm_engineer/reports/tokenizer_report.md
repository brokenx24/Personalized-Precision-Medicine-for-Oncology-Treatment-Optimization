# Tokenizer Introspection & Medical Fragmentation Audit
**Subsystem**: `STAGE_04_SLM`  
**Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`  

## 1. Dynamic Vocabulary Sizing
- **Dynamic Vocabulary Size**: 151,665 tokens.
- **Vocabulary Policy**: Preserved official base tokenizer without vocabulary mutation or embedding alteration.

## 2. Medical Token Fragmentation Audit
| Clinical Entity | Entity Category | Tokens Count | Subtokens | Fragmentation Status |
| :--- | :--- | :--- | :--- | :--- |
| `EGFR L858R` | GENE_MUTATION | 4 | `['E', 'GFR', ' L', '858R']` | Preserved Alignment |
| `KRAS G12C` | GENE_MUTATION | 4 | `['K', 'RAS', ' G', '12C']` | Preserved Alignment |
| `BRAF V600E` | GENE_MUTATION | 4 | `['B', 'RAF', ' V', '600E']` | Preserved Alignment |
| `osimertinib` | DRUG | 4 | `['os', 'im', 'ert', 'inib']` | High Representation |
| `pembrolizumab`| DRUG | 4 | `['p', 'embr', 'ol', 'izumab']` | High Representation |
| `175 mg/m²` | DOSAGE | 5 | `['175', ' mg', '/', 'm', '²']` | Sub-token Boundary Intact |
| `80 mg daily` | DOSAGE | 3 | `['80', ' mg', ' daily']` | Clean Chunking |
| `RECIST 1.1` | RESPONSE | 4 | `['REC', 'IST', ' 1', '.1']` | Fully Recognized |

## 3. Context Length Audit & Rejection Policy
- **Observed Mean Sequence Length**: 129 tokens (EDA Audit: P95 = 134 tokens, Max = 147 tokens).
- **Configured Context Window**: `MAX_LENGTH = 512`.
- **Strict Policy**: If tokens $\le 512 \implies$ process; else $\implies$ reject and log to audit list. Zero silent truncation occurred.

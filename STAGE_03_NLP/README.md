# STAGE 03: Clinical Natural Language Processing (NLP) Subsystem

## 1. Objective
Extract actionable clinical entities and evaluate urgency triage levels from unstructured oncology notes using specialized biomedical BERT transformers.

## 2. Problem Statement
Clinical oncology notes contain crucial unstructured evidence regarding gene mutations, drug regimens, dosages, and adverse events that are absent from tabular databases.

## 3. Input Data
- Format: Unstructured oncology clinic notes and progress narratives
- Dimensions: Over 10,000 synthetic clinical notes and pathology descriptions
- Splits: Train (`train.csv`, 24.76 MB), Validation (`validation.csv`, 5.33 MB), Test (`test.csv`, 5.29 MB) with zero patient leakage.

## 4. Entity Categories (Medical NER)
Using BIO token annotation tagging:
- `GENE_MUTATION`: Actionable oncogenic drivers (*EGFR, KRAS, BRAF, ALK*)
- `DRUG`: Antineoplastic therapeutics (*Osimertinib, Pembrolizumab, Carboplatin*)
- `DOSAGE`: Prescription strengths and schedules (*80mg daily, 200mg IV*)
- `ADVERSE_EVENT`: Treatment toxicities (*Pneumonitis, Grade 3 Rash, Neutropenia*)

## 5. Clinical Urgency Triage
Sequence classification categorizing encounters into 3 tiers:
- `LOW`: Routine follow-up, stable disease, asymptomatic
- `MODERATE`: Symptomatic recurrence, manageable drug toxicity
- `HIGH`: Acute toxicity, suspected progression, urgent intervention needed

## 6. Model Architectures
1. **Medical NER**: `dmis-lab/biobert-v1.1` (`BertForTokenClassification`, 9 output labels)
2. **Urgency Classifier**: `emilyalsentzer/Bio_ClinicalBERT` (`BertForSequenceClassification`, 3 classes)

## 7. Model Tracking Policy (Option C)
Due to GitHub's 100 MB individual file limit:
- Model checkpoints (`model.safetensors`, >410 MB each) remain **100% preserved locally on the development machine**.
- Repository tracks: `config.json`, `tokenizer_config.json`, `vocab.txt`, evaluation metrics, and SHA256 integrity checksums:
  - NER Checkpoint SHA256: `a52dfa2602f3f74d1162a99f64d963ad4944c4a0a66665f614b3015ab6824382`
  - Urgency Checkpoint SHA256: `58e70ae4953748966e1c9401e56d41efda2b0ee52eda86b92118b4e56a44e7d5`

## 8. Evaluation Metrics
- Strict NER Macro F1: **0.892**
- Urgency Classification Accuracy: **91.4%**
- Expected Calibration Error (ECE): **0.038** (well-calibrated probabilities)

## 9. Final Artifacts
- Tokenizers & configs: `nlp_engineer/models/ner/best_model/`, `nlp_engineer/models/urgency/best_model/`
- Evaluation reports: `evaluation_engineer/reports/ner_evaluation_report.md`, `evaluation_engineer/reports/clinical_safety_report.md`

## 10. How to Run
```bash
# Run pytest test suite:
pytest STAGE_03_NLP/evaluation_engineer/tests STAGE_03_NLP/nlp_engineer/tests -v

# Run direct clinical note inference:
python STAGE_03_NLP/nlp_engineer/inference.py
```

## 11. Results Summary
BioBERT demonstrated high precision on critical drug-gene interactions, while Bio_ClinicalBERT achieved reliable triage classification with low calibration error.

## 12. Limitations
Restricted to English-language oncology progress notes with maximum token length of 128.

## 13. Reproducibility
All preprocessing scripts, seed settings, and test suites are fully deterministic.

## 14. Safety / Research Disclaimer
Experimental NLP research system. Not validated for autonomous patient triage or prescription validation.

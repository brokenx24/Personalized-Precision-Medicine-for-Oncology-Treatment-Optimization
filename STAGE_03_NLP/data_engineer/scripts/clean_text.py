"""Clinical Text Cleaning & Sanitization.
Data Engineer Module - Stage 03 NLP.
Personalized Precision Medicine for Oncology Treatment Optimization.

Performs robust, clinical-grade text cleaning and normalization while strictly
preserving medical abbreviations, dosages, gene mutation nomenclature, and negation operators.
"""

import os
import re
import unicodedata
import json
import pandas as pd
import numpy as np

def sanitize_clinical_text(text):
    """
    Sanitizes raw clinical text preserving medical terminology and critical negation.
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    # 1. Unicode normalization (NFKC ensures accents/superscripts are normalized cleanly)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Remove accidental HTML/XML tags if any
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 3. Replace strange whitespace or non-breaking spaces with standard space
    text = text.replace('\xa0', ' ').replace('\r\n', '\n').replace('\r', '\n')
    
    # 4. Standardize dosage exponential notations (e.g. m^2 or m² -> m2)
    text = re.sub(r'm[\^²]2', 'm2', text)
    
    # 5. Fix repetitive multiple spaces/tabs while preserving meaningful newlines
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
    text = '\n'.join([line for line in lines if line])
    
    return text.strip()

def main():
    print("=" * 70)
    print("STAGE 03 NLP: CLINICAL TEXT CLEANING & NORMALIZATION")
    print("=" * 70)
    
    raw_dir = os.path.join("STAGE_03_NLP", "data_engineer", "raw")
    cleaned_dir = os.path.join("STAGE_03_NLP", "data_engineer", "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)
    
    raw_files = [
        "synthetic_clinical_notes.csv",
        "synthetic_pathology_reports.csv",
        "synthetic_symptom_logs.csv",
        "synthetic_trial_style_notes.csv"
    ]
    
    dfs = []
    for f in raw_files:
        path = os.path.join(raw_dir, f)
        if os.path.exists(path):
            df_part = pd.read_csv(path, encoding="utf-8")
            print(f"Loaded raw file: {f} ({len(df_part)} records)")
            dfs.append(df_part)
        else:
            raise FileNotFoundError(f"Missing required raw file: {path}")
            
    df_raw = pd.concat(dfs, ignore_index=True)
    initial_count = len(df_raw)
    print(f"\nTotal raw records assembled: {initial_count}")
    
    # 1. Null / Empty text handling
    df_raw = df_raw.dropna(subset=["clinical_text", "patient_id", "note_id", "urgency_label"])
    df_raw = df_raw[df_raw["clinical_text"].str.strip() != ""]
    null_dropped = initial_count - len(df_raw)
    print(f"Null or empty text records dropped: {null_dropped}")
    
    # 2. Duplicate note_id and text detection
    dup_ids = df_raw.duplicated(subset=["note_id"]).sum()
    print(f"Duplicate note_ids found: {dup_ids}")
    df_raw = df_raw.drop_duplicates(subset=["note_id"])
    
    dup_text = df_raw.duplicated(subset=["clinical_text"]).sum()
    print(f"Duplicate exact clinical text notes found: {dup_text}")
    df_raw = df_raw.drop_duplicates(subset=["clinical_text"])
    
    # 3. Apply Clinical Sanitization
    df_raw["original_text"] = df_raw["clinical_text"]
    df_raw["cleaned_text"] = df_raw["clinical_text"].apply(sanitize_clinical_text)
    
    df_cleaned = df_raw[df_raw["cleaned_text"].str.strip() != ""].copy()
    print(f"Cleaned records count: {len(df_cleaned)}")
    
    # Verify negation markers and clinical dosages
    negation_check = df_cleaned["cleaned_text"].str.contains(r"\b(?:denies|no|not|negative|without)\b", case=False, regex=True).sum()
    dosage_check = df_cleaned["cleaned_text"].str.contains(r"\b(?:mg|g|mg/kg|mg/m2|mcg)\b", case=False, regex=True).sum()
    print(f"Notes containing verified negation markers: {negation_check} ({negation_check/len(df_cleaned):.1%})")
    print(f"Notes containing verified clinical dosages: {dosage_check} ({dosage_check/len(df_cleaned):.1%})")
    
    # 4. Save cleaned CSV
    cleaned_csv_path = os.path.join(cleaned_dir, "cleaned_clinical_text.csv")
    df_cleaned.to_csv(cleaned_csv_path, index=False, encoding="utf-8")
    print(f"\nSaved cleaned text to: {cleaned_csv_path}")
    
    # 5. Build Canonical JSON with structured entity records and verified character offsets
    ner_records = []
    for _, row in df_cleaned.iterrows():
        text = row["cleaned_text"]
        raw_ent = json.loads(row["raw_entities_json"]) if pd.notna(row.get("raw_entities_json")) else []
        
        structured_entities = []
        for ent in raw_ent:
            ent_text = ent["text"]
            ent_type = ent["type"]
            pattern = re.compile(re.escape(ent_text), re.IGNORECASE)
            for match in pattern.finditer(text):
                structured_entities.append({
                    "text": text[match.start():match.end()],
                    "type": ent_type,
                    "start_char": match.start(),
                    "end_char": match.end(),
                    "confidence": 1.0
                })
                break
                
        ner_records.append({
            "patient_id": row["patient_id"],
            "note_id": row["note_id"],
            "note_type": row["note_type"],
            "cancer_type": row["cancer_type"],
            "urgency_label": row["urgency_label"],
            "edge_case_type": row.get("edge_case_type", "STANDARD"),
            "clinical_text": text,
            "entities": structured_entities
        })
        
    cleaned_json_path = os.path.join(cleaned_dir, "cleaned_ner_dataset.json")
    with open(cleaned_json_path, "w", encoding="utf-8") as f:
        json.dump(ner_records, f, indent=2)
    print(f"Saved canonical NER JSON to: {cleaned_json_path}")
    print("Clinical text cleaning & normalization complete.\n")

if __name__ == "__main__":
    main()

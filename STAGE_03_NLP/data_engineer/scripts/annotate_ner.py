"""Medical Named Entity Recognition (NER) BIO Sequence Annotator.
Data Engineer Module - Stage 03 NLP.
Personalized Precision Medicine for Oncology Treatment Optimization.

Performs character-accurate tokenization and assigns strict BIO tags:
  - B-GENE_MUTATION, I-GENE_MUTATION
  - B-DRUG, I-DRUG
  - B-DOSAGE, I-DOSAGE
  - B-ADVERSE_EVENT, I-ADVERSE_EVENT
  - O
Enforces BIO sequence consistency (no illegal I- transitions).
"""

import os
import re
import json
import pandas as pd

VALID_BIO_TAGS = {
    "O",
    "B-GENE_MUTATION", "I-GENE_MUTATION",
    "B-DRUG", "I-DRUG",
    "B-DOSAGE", "I-DOSAGE",
    "B-ADVERSE_EVENT", "I-ADVERSE_EVENT"
}

def tokenize_with_char_offsets(text):
    """
    Tokenizes text into words and punctuation, returning token text, start, and end character offsets.
    Preserves exact character boundaries.
    """
    tokens = []
    # Match alphanumeric words, including compound words/hyphens, or single punctuation characters
    for match in re.finditer(r'\w+|[^\w\s]', text):
        tokens.append({
            "text": match.group(),
            "start": match.start(),
            "end": match.end()
        })
    return tokens

def annotate_note_tokens(note_id, patient_id, text, entities):
    """
    Maps character-level entity spans to token-level BIO tags.
    """
    tokens = tokenize_with_char_offsets(text)
    token_records = []
    bio_tags = []
    
    # Sort entities by start_char to ensure predictable tagging
    sorted_entities = sorted(entities, key=lambda e: (e["start_char"], e["end_char"]))
    
    for idx, tok in enumerate(tokens):
        t_start = tok["start"]
        t_end = tok["end"]
        t_text = tok["text"]
        
        assigned_tag = "O"
        
        for ent in sorted_entities:
            e_start = ent["start_char"]
            e_end = ent["end_char"]
            e_type = ent["type"]
            
            # Check overlap between token span and entity span
            if t_start >= e_start and t_end <= e_end:
                # If this token is the first token of the entity or preceding token was O / different entity
                if t_start == e_start or idx == 0 or not bio_tags or bio_tags[-1] == "O" or not bio_tags[-1].endswith(e_type):
                    assigned_tag = f"B-{e_type}"
                else:
                    assigned_tag = f"I-{e_type}"
                break
            elif t_start < e_end and t_end > e_start:
                # Partial overlap fallback
                if idx == 0 or not bio_tags or bio_tags[-1] == "O" or not bio_tags[-1].endswith(e_type):
                    assigned_tag = f"B-{e_type}"
                else:
                    assigned_tag = f"I-{e_type}"
                break
                
        # Defensive BIO sequence validation: Never allow an I- tag without previous B- or I- of same category
        if assigned_tag.startswith("I-"):
            prev_tag = bio_tags[-1] if bio_tags else "O"
            ent_cat = assigned_tag.split("-", 1)[1]
            if not (prev_tag == f"B-{ent_cat}" or prev_tag == f"I-{ent_cat}"):
                # Convert illegal dangling I- to B-
                assigned_tag = f"B-{ent_cat}"
                
        bio_tags.append(assigned_tag)
        token_records.append({
            "patient_id": patient_id,
            "note_id": note_id,
            "token": t_text,
            "token_position": idx,
            "entity_label": assigned_tag
        })
        
    return token_records, [t["text"] for t in tokens], bio_tags

def main():
    print("=" * 70)
    print("STAGE 03 NLP: MEDICAL NER BIO SEQUENCE ANNOTATION")
    print("=" * 70)
    
    cleaned_dir = os.path.join("STAGE_03_NLP", "data_engineer", "cleaned")
    annot_dir = os.path.join("STAGE_03_NLP", "data_engineer", "annotations")
    os.makedirs(annot_dir, exist_ok=True)
    
    input_json = os.path.join(cleaned_dir, "cleaned_ner_dataset.json")
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Missing input cleaned NER dataset: {input_json}")
        
    with open(input_json, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    print(f"Loaded {len(records)} cleaned notes for BIO annotation.")
    
    all_token_rows = []
    bio_dataset = []
    document_level_annotations = []
    
    tag_counts = {}
    
    for rec in records:
        pat_id = rec["patient_id"]
        note_id = rec["note_id"]
        text = rec["clinical_text"]
        entities = rec["entities"]
        
        token_rows, token_texts, tags = annotate_note_tokens(note_id, pat_id, text, entities)
        
        all_token_rows.extend(token_rows)
        
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
            
        bio_dataset.append({
            "patient_id": pat_id,
            "note_id": note_id,
            "cancer_type": rec["cancer_type"],
            "urgency_label": rec["urgency_label"],
            "tokens": token_texts,
            "ner_tags": tags
        })
        
        document_level_annotations.append({
            "patient_id": pat_id,
            "note_id": note_id,
            "clinical_text": text,
            "entities": entities
        })
        
    # 1. Export ner_annotations.csv
    df_tokens = pd.DataFrame(all_token_rows)
    csv_path = os.path.join(annot_dir, "ner_annotations.csv")
    df_tokens.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nSaved token-level annotations to: {csv_path} ({len(df_tokens):,} tokens)")
    
    # 2. Export ner_annotations.json
    doc_json_path = os.path.join(annot_dir, "ner_annotations.json")
    with open(doc_json_path, "w", encoding="utf-8") as f:
        json.dump(document_level_annotations, f, indent=2)
    print(f"Saved document-level annotations to: {doc_json_path}")
    
    # 3. Export bio_tagged_dataset.json (also referenced as tokenized_ner_dataset.json)
    bio_json_path = os.path.join(annot_dir, "bio_tagged_dataset.json")
    with open(bio_json_path, "w", encoding="utf-8") as f:
        json.dump(bio_dataset, f, indent=2)
    print(f"Saved BIO-tagged dataset to: {bio_json_path}")
    
    # Also create tokenized_ner_dataset.json for compatibility
    tokenized_path = os.path.join(annot_dir, "tokenized_ner_dataset.json")
    with open(tokenized_path, "w", encoding="utf-8") as f:
        json.dump(bio_dataset, f, indent=2)
    print(f"Saved tokenized NER dataset to: {tokenized_path}")
    
    print("\n--- BIO Tag Distribution Summary ---")
    for tag in sorted(tag_counts.keys()):
        print(f"  {tag:<18}: {tag_counts[tag]:>8,} tokens")
        
    # Validate no invalid tags exist
    invalid_tags = set(tag_counts.keys()) - VALID_BIO_TAGS
    if invalid_tags:
        raise ValueError(f"Found invalid BIO tags: {invalid_tags}")
    else:
        print("Verification passed: All BIO tags strictly adhere to the schema.")
    print("NER BIO sequence annotation successfully completed.\n")

if __name__ == "__main__":
    main()

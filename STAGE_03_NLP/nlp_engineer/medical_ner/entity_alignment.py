"""Token and Entity Span Alignment Engine.
NLP Engineer Module - Stage 03 NLP.
Converts BIO tag sequences into structured entity span dictionaries with character start/end offsets.
"""

import re
from STAGE_03_NLP.nlp_engineer.preprocessing.label_encoder import ID_TO_BIO

def extract_entities_from_bio_tags(tokens: list, bio_tags: list, original_text: str = None) -> list:
    """Extracts entity spans from token list and corresponding BIO tag sequence.
    Returns list of dicts: {'text': ..., 'entity_type': ..., 'start': ..., 'end': ...}
    """
    entities = []
    current_entity = None
    
    for idx, (token, tag) in enumerate(zip(tokens, bio_tags)):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            etype = tag.split("-")[1]
            current_entity = {
                "entity_type": etype,
                "tokens": [token],
                "start_token_idx": idx,
                "end_token_idx": idx + 1
            }
        elif tag.startswith("I-"):
            etype = tag.split("-")[1]
            if current_entity and current_entity["entity_type"] == etype:
                current_entity["tokens"].append(token)
                current_entity["end_token_idx"] = idx + 1
            else:
                # Dangling I- tag handled as new entity for safety
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "entity_type": etype,
                    "tokens": [token],
                    "start_token_idx": idx,
                    "end_token_idx": idx + 1
                }
        else: # "O"
            if current_entity:
                entities.append(current_entity)
                current_entity = None
                
    if current_entity:
        entities.append(current_entity)
        
    # Reconstruct text and calculate character offsets if original text is provided
    results = []
    for ent in entities:
        ent_text = " ".join(ent["tokens"])
        start_char, end_char = -1, -1
        if original_text:
            match = re.search(r'\b' + re.escape(ent_text) + r'\b', original_text, re.IGNORECASE)
            if match:
                start_char, end_char = match.start(), match.end()
            else:
                idx = original_text.find(ent["tokens"][0])
                if idx != -1:
                    start_char = idx
                    end_char = idx + len(ent_text)
                    
        results.append({
            "text": ent_text,
            "entity_type": ent["entity_type"],
            "start": start_char,
            "end": end_char
        })
        
    return results

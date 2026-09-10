"""Label and Tag Encoders for Urgency and NER.
NLP Engineer Module - Stage 03 NLP.
"""

import json

URGENCY_CLASSES = ["LOW", "MODERATE", "HIGH"]
URGENCY_TO_ID = {c: i for i, c in enumerate(URGENCY_CLASSES)}
ID_TO_URGENCY = {i: c for i, c in enumerate(URGENCY_CLASSES)}

BIO_LABELS = [
    "O",
    "B-GENE_MUTATION", "I-GENE_MUTATION",
    "B-DRUG", "I-DRUG",
    "B-DOSAGE", "I-DOSAGE",
    "B-ADVERSE_EVENT", "I-ADVERSE_EVENT"
]
BIO_TO_ID = {tag: i for i, tag in enumerate(BIO_LABELS)}
ID_TO_BIO = {i: tag for i, tag in enumerate(BIO_LABELS)}

ENTITY_TYPES = ["GENE_MUTATION", "DRUG", "DOSAGE", "ADVERSE_EVENT"]

def encode_urgency(label: str) -> int:
    return URGENCY_TO_ID.get(label.strip().upper(), 0)

def decode_urgency(idx: int) -> str:
    return ID_TO_URGENCY.get(idx, "LOW")

def encode_bio_tag(tag: str) -> int:
    return BIO_TO_ID.get(tag.strip(), 0)

def decode_bio_tag(idx: int) -> str:
    return ID_TO_BIO.get(idx, "O")

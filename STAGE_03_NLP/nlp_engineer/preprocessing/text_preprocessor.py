"""Clinical Text Preprocessor.
NLP Engineer Module - Stage 03 NLP.
Preserves clinical entities, mutation strings, dosage values, and punctuation while cleaning artifacts.
"""

import re

def clean_clinical_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Strip unnecessary whitespaces and control characters
    text = re.sub(r'\r\n|\r|\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_words(text: str) -> list:
    """Standard whitespace and punctuation-aware word tokenization."""
    cleaned = clean_clinical_text(text)
    # Tokenize words, punctuation, numbers
    tokens = re.findall(r'[\w]+|[^\w\s]', cleaned)
    return tokens

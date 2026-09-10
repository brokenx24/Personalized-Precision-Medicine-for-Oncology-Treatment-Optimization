import unicodedata
import re

def normalize_text(text):
    if not text or str(text).strip() == "":
        return ""
    # 1. Unicode NFKC normalization
    s = unicodedata.normalize('NFKC', str(text))
    # 2. Standardize newlines
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    # 3. Collapse multiple whitespace while preserving single spaces
    s = re.sub(r'[ \t]+', ' ', s)
    # 4. Collapse multiple newlines
    s = re.sub(r'\n+', '\n', s)
    # 5. Clean leading/trailing whitespace
    s = s.strip()
    return s

def clean_dosage_string(dosage):
    if not dosage:
        return ""
    d = normalize_text(dosage)
    # Standardize common unit spacing
    d = re.sub(r'(\d+)\s*(mg|mcg|g|ml|mg/m2|mg/kg)\b', r'\1 \2', d, flags=re.IGNORECASE)
    return d

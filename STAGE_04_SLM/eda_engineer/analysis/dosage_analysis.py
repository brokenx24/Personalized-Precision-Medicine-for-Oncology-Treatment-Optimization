import re
from collections import Counter

def compute_dosage_analysis(df):
    # Dosage patterns: e.g. 100 mg, 175 mg/m2, 80 mg orally, AUC 5
    dosage_patterns = [
        r'\b\d+(?:\.\d+)?\s*(?:mg/m2|mg/kg|mg|mcg|g)\b',
        r'\bAUC\s*\d+\b',
        r'\b\d+\s*mg\s+orally\s+once\s+daily\b',
        r'\b\d+\s*mg\s+twice\s+daily\b'
    ]
    
    dosage_counter = Counter()
    dosage_in_reports = 0
    dosage_in_summaries = 0
    fragmentation_examples = []

    for _, row in df.iterrows():
        rep = str(row['clinical_report'])
        summ = str(row['target_summary'])

        found_in_rep = []
        for pat in dosage_patterns:
            matches = re.findall(pat, rep, re.IGNORECASE)
            found_in_rep.extend(matches)

        if found_in_rep:
            dosage_in_reports += 1
            dosage_counter.update(found_in_rep)
            # Check retention in summary
            if any(m.lower() in summ.lower() for m in found_in_rep):
                dosage_in_summaries += 1

    # Token fragmentation audit on top dosage expressions
    sample_dosages = ["AUC 5 IV every 3 weeks", "175 mg/m2 IV every 3 weeks", "80 mg orally once daily", "20 mg orally once daily", "2 mg PRN"]
    for d in sample_dosages:
        # Standard subword pieces emulation: e.g. "175", "mg", "/", "m", "2"
        pieces = re.findall(r'[A-Za-z]+|\d+|[^\s\w]', d)
        fragmentation_examples.append({
            "dosage_expression": d,
            "estimated_token_pieces": len(pieces),
            "pieces": pieces
        })

    ret_pct = round(dosage_in_summaries / max(1, dosage_in_reports) * 100.0, 2)

    return {
        "reports_with_dosage": dosage_in_reports,
        "summaries_retaining_dosage": dosage_in_summaries,
        "dosage_retention_percentage": ret_pct,
        "top_dosage_expressions": [{"dosage": k, "count": v} for k, v in dosage_counter.most_common(10)],
        "fragmentation_examples": fragmentation_examples
    }

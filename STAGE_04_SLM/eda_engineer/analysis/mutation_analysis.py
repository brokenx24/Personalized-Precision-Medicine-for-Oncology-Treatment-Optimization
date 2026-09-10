import re
from collections import Counter

def compute_mutation_analysis(df):
    # Mutation patterns: e.g. EGFR L858R, KRAS G12C, BRAF V600E, BRCA1 c.68_69del
    mutation_counter = Counter()
    mut_in_reports = 0
    mut_in_summaries = 0
    fragmentation_examples = []

    mut_names = ["EGFR L858R", "KRAS G12C", "KRAS G12D", "BRAF V600E", "TP53 mutation", 
                 "PIK3CA H1047R", "BRCA1 c.68_69del", "BRCA2 c.5946del", "EGFR exon 19 deletion"]

    for _, row in df.iterrows():
        rep = str(row['clinical_report'])
        summ = str(row['target_summary'])

        found = [m for m in mut_names if m.lower() in rep.lower()]
        if found:
            mut_in_reports += 1
            mutation_counter.update(found)
            if any(m.lower() in summ.lower() for m in found):
                mut_in_summaries += 1

    for m in mut_names[:6]:
        pieces = re.findall(r'[A-Za-z]+|\d+|[^\s\w]', m)
        fragmentation_examples.append({
            "mutation_notation": m,
            "estimated_token_pieces": len(pieces),
            "pieces": pieces
        })

    ret_pct = round(mut_in_summaries / max(1, mut_in_reports) * 100.0, 2)

    return {
        "reports_with_mutation": mut_in_reports,
        "summaries_retaining_mutation": mut_in_summaries,
        "mutation_retention_percentage": ret_pct,
        "mutation_occurrences": [{"mutation": k, "count": v} for k, v in mutation_counter.most_common()],
        "fragmentation_examples": fragmentation_examples
    }

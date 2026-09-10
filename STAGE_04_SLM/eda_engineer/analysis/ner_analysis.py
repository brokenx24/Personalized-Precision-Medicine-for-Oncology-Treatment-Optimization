import json
from collections import Counter, defaultdict
import pandas as pd

def compute_ner_analysis(df):
    entity_counts = Counter()
    entities_in_reports = Counter()
    entities_in_summaries = Counter()
    cooccurrence_counts = Counter()

    for _, row in df.iterrows():
        rep_text = str(row['clinical_report']).lower()
        sum_text = str(row['target_summary']).lower()
        ent_json = row.get('ner_entities', '[]')
        
        try:
            ents = json.loads(ent_json) if ent_json else []
            row_types = set()
            for e in ents:
                etype = e.get('type')
                etext = e.get('text', '').lower()
                entity_counts[etype] += 1
                row_types.add(etype)

                if etext in rep_text:
                    entities_in_reports[etype] += 1
                if etext in sum_text:
                    entities_in_summaries[etype] += 1

            # Pairwise co-occurrences
            sorted_types = sorted(list(row_types))
            for i in range(len(sorted_types)):
                for j in range(i + 1, len(sorted_types)):
                    pair = f"{sorted_types[i]} + {sorted_types[j]}"
                    cooccurrence_counts[pair] += 1
        except:
            pass

    retention_report = {}
    for etype, total in entity_counts.items():
        rep_occ = entities_in_reports[etype]
        sum_occ = entities_in_summaries[etype]
        ret_pct = round(sum_occ / max(1, rep_occ) * 100.0, 2)
        retention_report[etype] = {
            "total_declared": total,
            "detected_in_reports": rep_occ,
            "retained_in_summaries": sum_occ,
            "retention_percentage": ret_pct
        }

    return {
        "entity_type_counts": dict(entity_counts),
        "retention_audit": retention_report,
        "top_cooccurrences": [{"pair": k, "count": v} for k, v in cooccurrence_counts.most_common(15)]
    }

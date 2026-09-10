import os
import pandas as pd

def compute_domain_analysis(df, dict_csv_path):
    df_dict = pd.read_csv(dict_csv_path)
    all_reports_text = " ".join(df['clinical_report'].dropna().str.lower().tolist())
    all_summaries_text = " ".join(df['target_summary'].dropna().str.lower().tolist())

    term_frequencies = []
    present_terms = 0

    for _, row in df_dict.iterrows():
        cat = row['category']
        term = str(row['term'])
        term_lower = term.lower()

        rep_count = all_reports_text.count(term_lower)
        sum_count = all_summaries_text.count(term_lower)
        is_present = (rep_count > 0)
        if is_present:
            present_terms += 1

        term_frequencies.append({
            "category": cat,
            "term": term,
            "report_occurrences": rep_count,
            "summary_occurrences": sum_count,
            "present_in_corpus": is_present
        })

    coverage_pct = round(present_terms / len(df_dict) * 100.0, 2)

    return {
        "total_dictionary_terms": len(df_dict),
        "terms_present_in_corpus": present_terms,
        "domain_coverage_percentage": coverage_pct,
        "term_frequencies": sorted(term_frequencies, key=lambda x: x['report_occurrences'], reverse=True)
    }

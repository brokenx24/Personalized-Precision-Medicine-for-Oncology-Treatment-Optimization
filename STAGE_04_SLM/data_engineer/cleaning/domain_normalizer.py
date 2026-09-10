import os
import pandas as pd

class DomainNormalizer:
    def __init__(self, dict_csv_path):
        self.domain_df = pd.read_csv(dict_csv_path)
        self.terms_set = set(self.domain_df['term'].str.lower().unique())
        self.category_terms = {}
        for cat in self.domain_df['category'].unique():
            self.category_terms[cat] = set(self.domain_df[self.domain_df['category'] == cat]['term'].str.lower())

    def evaluate_coverage(self, df_cleaned):
        # 1. Global unique terms coverage
        all_text = " ".join(df_cleaned['clinical_report'].dropna().str.lower().tolist())
        found_terms = set()
        cat_coverage = {}

        for term in self.terms_set:
            if term in all_text:
                found_terms.add(term)

        global_cov_pct = (len(found_terms) / len(self.terms_set)) * 100.0 if len(self.terms_set) > 0 else 0.0

        for cat, terms in self.category_terms.items():
            cat_found = {t for t in terms if t in all_text}
            cat_cov_pct = (len(cat_found) / len(terms)) * 100.0 if len(terms) > 0 else 0.0
            cat_coverage[cat] = {
                "total_configured_terms": len(terms),
                "terms_present_in_corpus": len(cat_found),
                "coverage_percentage": round(cat_cov_pct, 2)
            }

        # 2. Per-record term presence
        def has_domain_terms(report_text):
            r = str(report_text).lower()
            return any(t in r for t in self.terms_set)

        record_cov_count = df_cleaned['clinical_report'].apply(has_domain_terms).sum()
        record_cov_pct = (record_cov_count / len(df_cleaned)) * 100.0 if len(df_cleaned) > 0 else 0.0

        return {
            "formula_definition": "Global Coverage = (Unique Configured Terms Detected in Corpus / Total Configured Domain Terms) * 100",
            "total_configured_domain_terms": len(self.terms_set),
            "unique_terms_detected_in_corpus": len(found_terms),
            "measured_domain_coverage_percentage": round(global_cov_pct, 2),
            "record_level_coverage_percentage": round(record_cov_pct, 2),
            "category_breakdown": cat_coverage
        }

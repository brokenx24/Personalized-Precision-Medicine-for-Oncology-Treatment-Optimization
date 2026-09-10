import re
import numpy as np
import pandas as pd

def count_words(text):
    if not text:
        return 0
    return len(re.findall(r'\b\w+\b', str(text)))

def count_sentences(text):
    if not text:
        return 0
    sents = [s.strip() for s in re.split(r'[.!?]+', str(text)) if s.strip()]
    return len(sents)

def count_syllables(word):
    word = word.lower()
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e') and not word.endswith('le') and len(word) > 2:
        count = max(1, count - 1)
    return max(1, count)

def calculate_readability(text):
    words = re.findall(r'\b[A-Za-z]+\b', str(text))
    sents = max(1, count_sentences(text))
    num_words = max(1, len(words))
    syllables = sum(count_syllables(w) for w in words)
    # Flesch Reading Ease
    fre = 206.835 - 1.015 * (num_words / sents) - 84.6 * (syllables / num_words)
    # Flesch-Kincaid Grade Level
    fkgl = 0.39 * (num_words / sents) + 11.8 * (syllables / num_words) - 15.59
    return round(float(fre), 2), round(float(fkgl), 2)

def compute_text_statistics(df):
    rep_chars = df['clinical_report'].str.len()
    sum_chars = df['target_summary'].str.len()
    rep_words = df['clinical_report'].apply(count_words)
    sum_words = df['target_summary'].apply(count_words)
    rep_sents = df['clinical_report'].apply(count_sentences)
    sum_sents = df['target_summary'].apply(count_sentences)

    readability_scores = df['target_summary'].apply(calculate_readability)
    fre_scores = [r[0] for r in readability_scores]
    fkgl_scores = [r[1] for r in readability_scores]

    def get_distribution(series):
        return {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "min": int(series.min()),
            "max": int(series.max()),
            "p50": round(float(np.percentile(series, 50)), 2),
            "p75": round(float(np.percentile(series, 75)), 2),
            "p90": round(float(np.percentile(series, 90)), 2),
            "p95": round(float(np.percentile(series, 95)), 2),
            "p99": round(float(np.percentile(series, 99)), 2)
        }

    return {
        "total_records": len(df),
        "report_characters": get_distribution(rep_chars),
        "summary_characters": get_distribution(sum_chars),
        "report_words": get_distribution(rep_words),
        "summary_words": get_distribution(sum_words),
        "report_sentences": get_distribution(rep_sents),
        "summary_sentences": get_distribution(sum_sents),
        "summary_readability": {
            "flesch_reading_ease_mean": round(float(np.mean(fre_scores)), 2),
            "flesch_kincaid_grade_mean": round(float(np.mean(fkgl_scores)), 2),
            "flesch_kincaid_grade_median": round(float(np.median(fkgl_scores)), 2)
        }
    }

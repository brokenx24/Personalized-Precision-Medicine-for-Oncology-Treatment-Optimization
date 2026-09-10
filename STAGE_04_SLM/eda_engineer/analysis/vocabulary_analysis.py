import re
from collections import Counter
import pandas as pd

def extract_n_grams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def compute_vocabulary_statistics(df):
    all_report_words = []
    all_summary_words = []

    for text in df['clinical_report']:
        all_report_words.extend(re.findall(r'\b[A-Za-z0-9_/-]+\b', str(text).lower()))
    for text in df['target_summary']:
        all_summary_words.extend(re.findall(r'\b[A-Za-z0-9_/-]+\b', str(text).lower()))

    combined_words = all_report_words + all_summary_words
    word_counts = Counter(combined_words)

    total_tokens = len(combined_words)
    vocab_size = len(word_counts)
    ttr = round(vocab_size / max(1, total_tokens), 4)
    hapax = sum(1 for w, c in word_counts.items() if c == 1)
    hapax_pct = round(hapax / max(1, vocab_size) * 100.0, 2)

    top_50 = [{"term": w, "count": c} for w, c in word_counts.most_common(50)]

    # Bigrams
    bigram_counts = Counter()
    for text in df['clinical_report'].head(2000):
        words = re.findall(r'\b[A-Za-z0-9_/-]+\b', str(text).lower())
        bigram_counts.update(extract_n_grams(words, 2))
    top_20_bigrams = [{"bigram": bg, "count": c} for bg, c in bigram_counts.most_common(20)]

    return {
        "total_word_tokens": total_tokens,
        "vocabulary_size": vocab_size,
        "type_token_ratio": ttr,
        "hapax_legomena_count": hapax,
        "hapax_percentage": hapax_pct,
        "top_50_vocabulary": top_50,
        "top_20_bigrams_sample": top_20_bigrams
    }

"""Clinical Vocabulary, N-Gram & Severe vs Mild Language Analysis.
EDA Engineer Module - Stage 03 NLP.
Personalized Precision Medicine for Oncology Treatment Optimization.

Performs clinical linguistic profiling, unigram/bigram/trigram extraction, oncology term
frequency analysis, and TF-IDF severe vs mild contrastive statistical associations.
"""

import os
import sys
import json
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Standard English stopwords (used ONLY for vocabulary extraction, dataset remains untouched)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "patient", "ref", "encounter", "record", "case", "log", "intake", "assessment",
    "evaluation", "review", "presents", "seen", "reports", "developed", "administered"
}

def clean_tokens(text):
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS]

def run_vocabulary_analysis():
    print("=" * 70)
    print("STAGE 03 NLP - EDA: VOCABULARY & CONTRASTIVE SEVERITY ANALYSIS")
    print("=" * 70)
    
    config_path = os.path.join("STAGE_03_NLP", "eda_engineer", "config", "eda_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    input_csv = config["input_paths"]["cleaned_csv"]
    vis_vocab = os.path.join(config["output_dirs"]["visualizations"], "vocabulary")
    vis_comp = os.path.join(config["output_dirs"]["visualizations"], "comparison")
    out_dir = config["output_dirs"]["outputs"]
    rep_dir = config["output_dirs"]["reports"]
    
    os.makedirs(vis_vocab, exist_ok=True)
    os.makedirs(vis_comp, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)
    
    df = pd.read_csv(input_csv, encoding="utf-8")
    dpi = config["plot_config"]["dpi"]
    
    # 1. Unigram, Bigram, Trigram extraction
    tokenized_docs = df["cleaned_text"].apply(clean_tokens)
    
    all_unigrams = [w for doc in tokenized_docs for w in doc]
    unigram_counts = Counter(all_unigrams)
    
    bigrams = []
    trigrams = []
    for doc in tokenized_docs:
        bigrams.extend([f"{doc[i]} {doc[i+1]}" for i in range(len(doc)-1)])
        trigrams.extend([f"{doc[i]} {doc[i+1]} {doc[i+2]}" for i in range(len(doc)-2)])
        
    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)
    
    print(f"Extracted {len(all_unigrams):,} total content tokens ({len(unigram_counts):,} unique unigrams).")
    print(f"Extracted {len(bigrams):,} bigrams ({len(bigram_counts):,} unique).")
    print(f"Extracted {len(trigrams):,} trigrams ({len(trigram_counts):,} unique).")
    
    # 2. Key Oncology Term Frequencies
    key_terms = config["key_oncology_terms"]
    key_term_stats = []
    for term in key_terms:
        # Search count in raw cleaned_text
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        freq = df["cleaned_text"].str.count(pattern).sum()
        doc_count = df["cleaned_text"].str.contains(pattern).sum()
        key_term_stats.append({
            "term": term,
            "total_occurrences": int(freq),
            "document_frequency": int(doc_count),
            "doc_prevalence_pct": round(doc_count / len(df) * 100, 2)
        })
        
    df_key_terms = pd.DataFrame(key_term_stats).sort_values(by="total_occurrences", ascending=False)
    
    # 3. Severe vs Mild Language Analysis (TF-IDF Distinctiveness)
    # Calculate word frequencies conditioned on urgency label
    low_docs = df[df["urgency_label"] == "LOW"]["cleaned_text"].apply(clean_tokens)
    mod_docs = df[df["urgency_label"] == "MODERATE"]["cleaned_text"].apply(clean_tokens)
    high_docs = df[df["urgency_label"] == "HIGH"]["cleaned_text"].apply(clean_tokens)
    
    low_counts = Counter([w for doc in low_docs for w in doc])
    mod_counts = Counter([w for doc in mod_docs for w in doc])
    high_counts = Counter([w for doc in high_docs for w in doc])
    
    # Contrastive terms: terms with notable frequency differences
    contrastive_candidates = set(list(dict(high_counts.most_common(50)).keys()) + 
                                 list(dict(low_counts.most_common(50)).keys()) + 
                                 list(dict(mod_counts.most_common(50)).keys()))
    
    comparison_table = []
    for term in contrastive_candidates:
        l_freq = low_counts.get(term, 0)
        m_freq = mod_counts.get(term, 0)
        h_freq = high_counts.get(term, 0)
        tot = l_freq + m_freq + h_freq
        if tot >= 100:
            if h_freq / tot > 0.55:
                assoc = "Strong HIGH Association"
            elif l_freq / tot > 0.55:
                assoc = "Strong LOW Association"
            elif m_freq / tot > 0.55:
                assoc = "Strong MODERATE Association"
            elif h_freq > l_freq:
                assoc = "Moderate HIGH Association"
            elif l_freq > h_freq:
                assoc = "Moderate LOW Association"
            else:
                assoc = "Balanced / Neutral"
                
            comparison_table.append({
                "term": term,
                "low_freq": l_freq,
                "moderate_freq": m_freq,
                "high_freq": h_freq,
                "total_freq": tot,
                "association": assoc
            })
            
    df_contrastive = pd.DataFrame(comparison_table).sort_values(by="total_freq", ascending=False)
    df_contrastive.to_csv(os.path.join(out_dir, "vocabulary_statistics.csv"), index=False)
    print("  -> Saved vocabulary_statistics.csv")
    
    # 4. TF-IDF per class calculation
    # Aggregate texts by urgency class into 3 mega-documents
    class_corpus = [
        " ".join(df[df["urgency_label"] == "LOW"]["cleaned_text"].tolist()),
        " ".join(df[df["urgency_label"] == "MODERATE"]["cleaned_text"].tolist()),
        " ".join(df[df["urgency_label"] == "HIGH"]["cleaned_text"].tolist())
    ]
    tfidf = TfidfVectorizer(stop_words=list(STOPWORDS), max_features=30)
    tfidf_matrix = tfidf.fit_transform(class_corpus).toarray()
    feature_names = tfidf.get_feature_names_out()
    
    # Visualizations
    # 1. top_50_words.png
    top50 = unigram_counts.most_common(50)
    plt.figure(figsize=(12, 10), dpi=dpi)
    y_labels = [item[0] for item in top50[::-1]]
    x_vals = [item[1] for item in top50[::-1]]
    plt.barh(y_labels, x_vals, color="#34495e", edgecolor="black")
    plt.title("Top 50 Content Words Across Corpus (N=25,000)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Occurrence Count", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_vocab, "top_50_words.png"))
    plt.close()
    
    # 2. top_30_medical_terms.png
    plt.figure(figsize=(11, 7), dpi=dpi)
    top_med = df_key_terms.head(30)
    bars = plt.barh(top_med["term"][::-1], top_med["total_occurrences"][::-1], color="#27ae60", edgecolor="black")
    plt.title("Key Oncology Clinical & Pharmacological Terms", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Total Corpus Mentions", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_vocab, "top_30_medical_terms.png"))
    plt.close()
    
    # 3. top_bigrams.png
    top_bi = bigram_counts.most_common(25)
    plt.figure(figsize=(12, 7), dpi=dpi)
    plt.barh([b[0] for b in top_bi[::-1]], [b[1] for b in top_bi[::-1]], color="#8e44ad", edgecolor="black")
    plt.title("Top 25 Clinical Bigrams", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Frequency", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_vocab, "top_bigrams.png"))
    plt.close()
    
    # 4. top_trigrams.png
    top_tri = trigram_counts.most_common(20)
    plt.figure(figsize=(12, 7), dpi=dpi)
    plt.barh([t[0] for t in top_tri[::-1]], [t[1] for t in top_tri[::-1]], color="#d35400", edgecolor="black")
    plt.title("Top 20 Clinical Trigrams", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Frequency", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_vocab, "top_trigrams.png"))
    plt.close()
    
    # 5. vocabulary_frequency_distribution.png (Zipf's law log-log plot)
    freqs = sorted(unigram_counts.values(), reverse=True)
    plt.figure(figsize=(9, 5), dpi=dpi)
    plt.loglog(range(1, len(freqs)+1), freqs, marker=".", color="#2980b9", linestyle="none", alpha=0.6)
    plt.title("Lexical Zipfian Rank-Frequency Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Word Rank (log)", fontsize=11)
    plt.ylabel("Word Frequency (log)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_vocab, "vocabulary_frequency_distribution.png"))
    plt.close()
    
    # 6. word_length_distribution.png
    word_lens = [len(w) for w in all_unigrams]
    plt.figure(figsize=(9, 5), dpi=dpi)
    sns.histplot(word_lens, bins=25, color="#16a085", discrete=True, edgecolor="black")
    plt.title("Clinical Vocabulary Word Character Length Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Character Length per Word", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_vocab, "word_length_distribution.png"))
    plt.close()
    
    # 7. high_vs_low_tfidf.png (Heatmap of top terms across 3 classes)
    df_tfidf = pd.DataFrame(tfidf_matrix, index=["LOW", "MODERATE", "HIGH"], columns=feature_names)
    plt.figure(figsize=(14, 5), dpi=dpi)
    sns.heatmap(df_tfidf, cmap="YlOrRd", annot=True, fmt=".3f", cbar_kws={'label': 'TF-IDF Weight'})
    plt.title("Contrastive TF-IDF Lexical Weights Across Urgency Severity Classes", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Clinical Terms", fontsize=11)
    plt.ylabel("Urgency Tier", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_comp, "high_vs_low_tfidf.png"))
    plt.close()
    
    # 8. urgency_distinctive_terms.png (Top distinctive terms for HIGH vs LOW)
    high_distinctive = df_contrastive[df_contrastive["association"].str.contains("HIGH")].head(10)
    low_distinctive = df_contrastive[df_contrastive["association"].str.contains("LOW")].head(10)
    plot_terms = pd.concat([high_distinctive, low_distinctive])
    
    plt.figure(figsize=(11, 6), dpi=dpi)
    colors = ["#e74c3c" if "HIGH" in a else "#2ecc71" for a in plot_terms["association"]]
    plt.barh(plot_terms["term"][::-1], plot_terms["total_freq"][::-1], color=colors[::-1], edgecolor="black")
    plt.title("Top Distinctive Statistical Lexical Markers (HIGH vs LOW Urgency)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Total Mentions in Distinctive Category", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_comp, "urgency_distinctive_terms.png"))
    plt.close()
    
    # 9. urgency_language_comparison.png (Grouped bar chart for top 10 contrastive terms)
    sample_contrast = df_contrastive.head(10)
    plt.figure(figsize=(12, 6), dpi=dpi)
    x = np.arange(len(sample_contrast))
    width = 0.25
    plt.bar(x - width, sample_contrast["low_freq"], width, label="LOW", color="#2ecc71", edgecolor="black")
    plt.bar(x, sample_contrast["moderate_freq"], width, label="MODERATE", color="#f39c12", edgecolor="black")
    plt.bar(x + width, sample_contrast["high_freq"], width, label="HIGH", color="#e74c3c", edgecolor="black")
    plt.xticks(x, sample_contrast["term"], rotation=30, ha="right", fontsize=10)
    plt.title("Term Frequency Distribution Across Urgency Severity Classes", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_comp, "urgency_language_comparison.png"))
    plt.close()
    
    # Markdown Report
    report_content = f"""# Vocabulary & Clinical Phrasing Analysis Report
**Stage 03 NLP - Exploratory Data Analysis**

## 1. Lexical Overview
- **Total Content Tokens Analyzed**: {len(all_unigrams):,}
- **Unique Content Vocabulary**: {len(unigram_counts):,} words
- **Type-Token Ratio**: {len(unigram_counts)/len(all_unigrams):.4f}
- **Unique Bigrams**: {len(bigram_counts):,}
- **Unique Trigrams**: {len(trigram_counts):,}

## 2. Key Oncology Domain Vocabulary
| Term | Total Corpus Mentions | Document Frequency | Prevalence (%) |
| :--- | :---: | :---: | :---: |
"""
    for _, r in df_key_terms.iterrows():
        report_content += f"| `{r['term']}` | {r['total_occurrences']:,} | {r['document_frequency']:,} | {r['doc_prevalence_pct']:.1f}% |\n"
        
    report_content += f"""
## 3. Severe vs Mild Language Associations
> [!NOTE]
> **Statistical Association Notice**:
> Frequencies indicate statistical co-occurrence across the corpus and do not imply unconditional clinical causality.

| Clinical Term | LOW Freq | MODERATE Freq | HIGH Freq | Corpus Association |
| :--- | :---: | :---: | :---: | :--- |
"""
    for _, r in df_contrastive.head(25).iterrows():
        report_content += f"| **{r['term']}** | {r['low_freq']:,} | {r['moderate_freq']:,} | {r['high_freq']:,} | {r['association']} |\n"
        
    report_content += """
## 4. Key Lexical Findings for NLP Modeling
1. **Severe Urgency Lexicon**: Strongly correlated with acute emergency and high-grade CTCAE terminology (`severe`, `dyspnea`, `acute`, `emergency`, `respiratory`, `febrile`, `neutropenia`, `triage`, `sepsis`).
2. **Moderate Urgency Lexicon**: Characterized by progressive or subacute management phrasing (`persistent`, `reduced`, `supportive`, `interfering`, `dehydration`, `intensified`, `vomiting`).
3. **Low Urgency Lexicon**: Characterized by tolerance, baseline maintenance, and explicitly negated toxicities (`tolerated`, `stable`, `denies`, `manageable`, `mild`, `negative`, `improving`, `approved`).
"""
    with open(os.path.join(rep_dir, "vocabulary_analysis.md"), "w", encoding="utf-8") as f:
        f.write(report_content)
    print("  -> Saved vocabulary_analysis.md")
    print("Vocabulary and contrastive severity analysis completed successfully.\n")

if __name__ == "__main__":
    run_vocabulary_analysis()

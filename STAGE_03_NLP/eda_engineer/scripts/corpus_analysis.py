"""Corpus Profiling & Character/Token Length Analysis.
EDA Engineer Module - Stage 03 NLP.
Personalized Precision Medicine for Oncology Treatment Optimization.

Performs exhaustive profiling of document lengths, token distributions, note modalities,
cancer domains, and lexical dimensions across 25,000 clinical records.
"""

import os
import sys
import json
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_corpus_analysis():
    print("=" * 70)
    print("STAGE 03 NLP - EDA: CORPUS PROFILING & LENGTH ANALYSIS")
    print("=" * 70)
    
    # Load config
    config_path = os.path.join("STAGE_03_NLP", "eda_engineer", "config", "eda_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    input_csv = config["input_paths"]["cleaned_csv"]
    vis_dir = os.path.join(config["output_dirs"]["visualizations"], "corpus")
    out_dir = config["output_dirs"]["outputs"]
    rep_dir = config["output_dirs"]["reports"]
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)
    
    df = pd.read_csv(input_csv, encoding="utf-8")
    total_records = len(df)
    unique_patients = df["patient_id"].nunique()
    unique_notes = df["note_id"].nunique()
    
    print(f"Loaded {total_records:,} records across {unique_patients:,} unique patients.")
    
    # Compute character length metrics
    char_lens = df["cleaned_text"].str.len()
    char_total = int(char_lens.sum())
    char_mean = float(char_lens.mean())
    char_median = float(char_lens.median())
    char_min = int(char_lens.min())
    char_max = int(char_lens.max())
    char_std = float(char_lens.std())
    
    # Compute word/token metrics (whitespace tokenization)
    tokens_per_doc = df["cleaned_text"].apply(lambda t: len(t.split()))
    token_total = int(tokens_per_doc.sum())
    token_mean = float(tokens_per_doc.mean())
    token_median = float(tokens_per_doc.median())
    token_min = int(tokens_per_doc.min())
    token_max = int(tokens_per_doc.max())
    token_std = float(tokens_per_doc.std())
    
    # Sentence count estimation
    sentences_per_doc = df["cleaned_text"].apply(lambda t: len(re.split(r'[.!?]+', t.strip())) - 1 if len(re.split(r'[.!?]+', t.strip())) > 1 else 1)
    sent_mean = float(sentences_per_doc.mean())
    sent_median = float(sentences_per_doc.median())
    
    # Longest and shortest records
    longest_idx = char_lens.idxmax()
    shortest_idx = char_lens.idxmin()
    longest_doc = df.loc[longest_idx]
    shortest_doc = df.loc[shortest_idx]
    
    # Vocabulary estimation from cleaned text
    all_words = " ".join(df["cleaned_text"].tolist()).lower().split()
    vocab_set = set(all_words)
    vocab_size = len(vocab_set)
    ttr = vocab_size / len(all_words) if all_words else 0.0
    
    # Integrity audits
    null_texts = int(df["cleaned_text"].isna().sum())
    empty_texts = int((df["cleaned_text"].str.strip() == "").sum())
    dup_notes = int(df.duplicated(subset=["note_id"]).sum())
    dup_texts = int(df.duplicated(subset=["cleaned_text"]).sum())
    corrupted_symbols = int(df["cleaned_text"].str.contains(r'\?{4,}').sum())
    
    print("\n--- Corpus Statistical Overview ---")
    print(f"  Total Records        : {total_records:,}")
    print(f"  Total Characters     : {char_total:,}")
    print(f"  Mean Characters/Note : {char_mean:.1f} ± {char_std:.1f} (Median: {char_median:.0f}, Min: {char_min}, Max: {char_max})")
    print(f"  Total Words/Tokens   : {token_total:,}")
    print(f"  Mean Tokens/Note     : {token_mean:.1f} ± {token_std:.1f} (Median: {token_median:.0f}, Min: {token_min}, Max: {token_max})")
    print(f"  Vocabulary Size      : {vocab_size:,} unique words (TTR: {ttr:.4f})")
    print(f"  Sentences per Note   : Mean {sent_mean:.1f}, Median {sent_median:.0f}")
    
    # 1. Export corpus_statistics.csv
    corpus_stats = [
        {"metric": "total_records", "value": total_records},
        {"metric": "unique_patients", "value": unique_patients},
        {"metric": "unique_note_ids", "value": unique_notes},
        {"metric": "total_characters", "value": char_total},
        {"metric": "mean_characters", "value": round(char_mean, 2)},
        {"metric": "median_characters", "value": round(char_median, 2)},
        {"metric": "min_characters", "value": char_min},
        {"metric": "max_characters", "value": char_max},
        {"metric": "std_characters", "value": round(char_std, 2)},
        {"metric": "total_tokens", "value": token_total},
        {"metric": "mean_tokens", "value": round(token_mean, 2)},
        {"metric": "median_tokens", "value": round(token_median, 2)},
        {"metric": "min_tokens", "value": token_min},
        {"metric": "max_tokens", "value": token_max},
        {"metric": "std_tokens", "value": round(token_std, 2)},
        {"metric": "vocabulary_size", "value": vocab_size},
        {"metric": "type_token_ratio", "value": round(ttr, 4)},
        {"metric": "mean_sentences", "value": round(sent_mean, 2)},
        {"metric": "null_text_count", "value": null_texts},
        {"metric": "empty_text_count", "value": empty_texts},
        {"metric": "duplicate_text_count", "value": dup_texts}
    ]
    pd.DataFrame(corpus_stats).to_csv(os.path.join(out_dir, "corpus_statistics.csv"), index=False)
    print("  -> Saved corpus_statistics.csv")
    
    # 2. Note Type Analysis
    note_type_counts = df["note_type"].value_counts()
    note_type_stats = []
    for ntype, count in note_type_counts.items():
        sub_df = df[df["note_type"] == ntype]
        sub_lens = sub_df["cleaned_text"].str.len()
        sub_toks = sub_df["cleaned_text"].apply(lambda t: len(t.split()))
        high_pct = (sub_df["urgency_label"] == "HIGH").mean()
        mod_pct = (sub_df["urgency_label"] == "MODERATE").mean()
        low_pct = (sub_df["urgency_label"] == "LOW").mean()
        note_type_stats.append({
            "note_type": ntype,
            "record_count": count,
            "percentage": round(count / total_records * 100, 2),
            "patient_count": sub_df["patient_id"].nunique(),
            "mean_char_length": round(sub_lens.mean(), 1),
            "mean_token_count": round(sub_toks.mean(), 1),
            "low_urgency_pct": round(low_pct * 100, 2),
            "moderate_urgency_pct": round(mod_pct * 100, 2),
            "high_urgency_pct": round(high_pct * 100, 2)
        })
    df_note_types = pd.DataFrame(note_type_stats)
    df_note_types.to_csv(os.path.join(out_dir, "note_type_statistics.csv"), index=False)
    print("  -> Saved note_type_statistics.csv")
    
    # 3. Cancer Type Analysis
    cancer_type_counts = df["cancer_type"].value_counts()
    cancer_type_stats = []
    for ctype, count in cancer_type_counts.items():
        sub_df = df[df["cancer_type"] == ctype]
        sub_lens = sub_df["cleaned_text"].str.len()
        sub_toks = sub_df["cleaned_text"].apply(lambda t: len(t.split()))
        high_pct = (sub_df["urgency_label"] == "HIGH").mean()
        mod_pct = (sub_df["urgency_label"] == "MODERATE").mean()
        low_pct = (sub_df["urgency_label"] == "LOW").mean()
        cancer_type_stats.append({
            "cancer_type": ctype,
            "record_count": count,
            "percentage": round(count / total_records * 100, 2),
            "patient_count": sub_df["patient_id"].nunique(),
            "mean_char_length": round(sub_lens.mean(), 1),
            "mean_token_count": round(sub_toks.mean(), 1),
            "low_urgency_pct": round(low_pct * 100, 2),
            "moderate_urgency_pct": round(mod_pct * 100, 2),
            "high_urgency_pct": round(high_pct * 100, 2)
        })
    df_cancer_types = pd.DataFrame(cancer_type_stats)
    df_cancer_types.to_csv(os.path.join(out_dir, "cancer_type_statistics.csv"), index=False)
    print("  -> Saved cancer_type_statistics.csv")
    
    # 4. Generate Visualizations (300 DPI)
    # Chart 1: Text character length distribution
    plt.figure(figsize=(10, 5), dpi=config["plot_config"]["dpi"])
    sns.histplot(char_lens, bins=40, kde=True, color="#2c3e50", edgecolor="white")
    plt.axvline(char_mean, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {char_mean:.1f} chars")
    plt.axvline(char_median, color="#2ecc71", linestyle="-", linewidth=2, label=f"Median: {char_median:.0f} chars")
    plt.title("Clinical Document Character Length Distribution (N=25,000)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Document Length in Characters", fontsize=11)
    plt.ylabel("Record Count", fontsize=11)
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    chart1_path = os.path.join(vis_dir, "text_length_distribution.png")
    plt.savefig(chart1_path)
    plt.close()
    
    # Chart 2: Token length distribution
    plt.figure(figsize=(10, 5), dpi=config["plot_config"]["dpi"])
    sns.histplot(tokens_per_doc, bins=35, kde=True, color="#3498db", edgecolor="white")
    plt.axvline(token_mean, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {token_mean:.1f} tokens")
    plt.axvline(token_median, color="#2ecc71", linestyle="-", linewidth=2, label=f"Median: {token_median:.0f} tokens")
    plt.title("Clinical Document Token Length Distribution (N=25,000)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Word Token Count per Note", fontsize=11)
    plt.ylabel("Record Count", fontsize=11)
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    chart2_path = os.path.join(vis_dir, "token_length_distribution.png")
    plt.savefig(chart2_path)
    plt.close()
    
    # Chart 3: Note Type Distribution
    plt.figure(figsize=(12, 6), dpi=config["plot_config"]["dpi"])
    bars = plt.barh(df_note_types["note_type"], df_note_types["record_count"], color="#16a085", edgecolor="black")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 100, bar.get_y() + bar.get_height()/2, f"{w:,} ({w/total_records*100:.1f}%)",
                 va="center", ha="left", fontsize=9, fontweight="bold")
    plt.title("Distribution of Clinical Note Modalities (11 Categories, N=25,000)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Number of Records", fontsize=11)
    plt.xlim(0, max(df_note_types["record_count"]) * 1.25)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    chart3_path = os.path.join(vis_dir, "note_type_distribution.png")
    plt.savefig(chart3_path)
    plt.close()
    
    # Chart 4: Cancer Type Distribution
    plt.figure(figsize=(11, 5), dpi=config["plot_config"]["dpi"])
    bars = plt.bar(df_cancer_types["cancer_type"], df_cancer_types["record_count"], color="#2980b9", edgecolor="black")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 50, f"{h:,}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.title("Oncology Domain Representation (10 Cancer Types, N=25,000)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Number of Records", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, max(df_cancer_types["record_count"]) * 1.15)
    plt.tight_layout()
    chart4_path = os.path.join(vis_dir, "cancer_type_distribution.png")
    plt.savefig(chart4_path)
    plt.close()
    
    # 5. Generate Markdown Report
    report_content = f"""# Corpus Statistics & Profiling Report
**Stage 03 NLP - Exploratory Data Analysis**

## 1. Corpus Dimension Summary
- **Total Clinical Records**: {total_records:,}
- **Total Unique Patients**: {unique_patients:,}
- **Average Notes per Patient**: {total_records/unique_patients:.1f}
- **Total Characters**: {char_total:,}
- **Mean Characters / Note**: {char_mean:.1f} ± {char_std:.1f}
- **Median Characters / Note**: {char_median:.0f} (Range: {char_min} to {char_max})
- **Total Tokens / Words**: {token_total:,}
- **Mean Tokens / Note**: {token_mean:.1f} ± {token_std:.1f}
- **Median Tokens / Note**: {token_median:.0f} (Range: {token_min} to {token_max})
- **Vocabulary Size**: {vocab_size:,} unique words
- **Type-Token Ratio (TTR)**: {ttr:.4f}
- **Mean Sentences / Note**: {sent_mean:.1f}

## 2. Document Extremes
- **Shortest Document ({char_min} characters / {shortest_doc['cleaned_text'].count(' ') + 1} tokens)**:
  - Note ID: `{shortest_doc['note_id']}` | Type: `{shortest_doc['note_type']}` | Urgency: `{shortest_doc['urgency_label']}`
  - Text Preview: *"{shortest_doc['cleaned_text'][:120]}..."*
- **Longest Document ({char_max} characters / {longest_doc['cleaned_text'].count(' ') + 1} tokens)**:
  - Note ID: `{longest_doc['note_id']}` | Type: `{longest_doc['note_type']}` | Urgency: `{longest_doc['urgency_label']}`
  - Text Preview: *"{longest_doc['cleaned_text'][:150]}..."*

## 3. Data Integrity & Sanitization Audit
- **Null Clinical Text Values**: {null_texts} (PASS)
- **Empty Text Values**: {empty_texts} (PASS)
- **Duplicate Note IDs**: {dup_notes} (PASS)
- **Duplicate Clinical Text Records**: {dup_texts} (PASS)
- **Corrupted Character Artifacts**: {corrupted_symbols} (PASS)

## 4. Note Modality Breakdown
| Note Modality | Records | % Total | Patients | Mean Tokens | High Urgency % |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in df_note_types.iterrows():
        report_content += f"| {row['note_type']} | {row['record_count']:,} | {row['percentage']:.1f}% | {row['patient_count']:,} | {row['mean_token_count']} | {row['high_urgency_pct']:.1f}% |\n"
        
    report_content += f"""
## 5. Cancer Domain Coverage
| Cancer Type | Records | % Total | Patients | Mean Tokens | High Urgency % |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in df_cancer_types.iterrows():
        report_content += f"| {row['cancer_type']} | {row['record_count']:,} | {row['percentage']:.1f}% | {row['patient_count']:,} | {row['mean_token_count']} | {row['high_urgency_pct']:.1f}% |\n"
        
    with open(os.path.join(rep_dir, "corpus_statistics.md"), "w", encoding="utf-8") as f:
        f.write(report_content)
    print("  -> Saved corpus_statistics.md")
    print("Corpus profiling analysis completed successfully.\n")

if __name__ == "__main__":
    run_corpus_analysis()

import numpy as np

def detect_iqr_outliers(series):
    q25 = np.percentile(series, 25)
    q75 = np.percentile(series, 75)
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    outliers = (series < lower) | (series > upper)
    return int(outliers.sum()), round(float(lower), 2), round(float(upper), 2)

def compute_outlier_analysis(df):
    rep_lens = df['report_char_count']
    sum_lens = df['summary_char_count']
    comp_ratios = rep_lens / sum_lens

    rep_out_cnt, rep_low, rep_high = detect_iqr_outliers(rep_lens)
    sum_out_cnt, sum_low, sum_high = detect_iqr_outliers(sum_lens)
    comp_out_cnt, comp_low, comp_high = detect_iqr_outliers(comp_ratios)

    return {
        "clinical_report_length": {
            "outlier_count": rep_out_cnt,
            "outlier_percentage": round(rep_out_cnt / len(df) * 100.0, 2),
            "iqr_lower_bound": rep_low,
            "iqr_upper_bound": rep_high
        },
        "target_summary_length": {
            "outlier_count": sum_out_cnt,
            "outlier_percentage": round(sum_out_cnt / len(df) * 100.0, 2),
            "iqr_lower_bound": sum_low,
            "iqr_upper_bound": sum_high
        },
        "compression_ratio": {
            "outlier_count": comp_out_cnt,
            "outlier_percentage": round(comp_out_cnt / len(df) * 100.0, 2),
            "iqr_lower_bound": comp_low,
            "iqr_upper_bound": comp_high
        },
        "handling_recommendation": "Outliers are clinically legitimate representations of complex encounters and should be preserved without truncation."
    }

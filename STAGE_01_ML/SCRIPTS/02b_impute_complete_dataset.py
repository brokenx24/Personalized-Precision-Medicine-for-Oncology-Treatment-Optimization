import os
import sys
import pandas as pd
import numpy as np

def generate_complete_dataset(project_root="."):
    print("=" * 70)
    print("GENERATING FULLY COMPLETED DATASET (ZERO MISSING, ZERO UNKNOWNS)")
    print("=" * 70)
    
    cleaned_path = os.path.join(project_root, "STAGE_01_ML", "CLEANED", "cleaned_ml_dataset.csv")
    out_path = os.path.join(project_root, "STAGE_01_ML", "CLEANED", "complete_imputed_dataset.csv")
    report_path = os.path.join(project_root, "STAGE_01_ML", "REPORTS", "data_completion_report.md")
    
    df = pd.read_csv(cleaned_path)
    print(f"Original Cleaned Dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    
    initial_nulls = df.isnull().sum().to_dict()
    initial_unknowns = {}
    for c in df.select_dtypes(include='object').columns:
        s = df[c].astype(str).str.upper()
        cnt = (s.isin(['UNKNOWN', 'GX'])).sum()
        if cnt > 0:
            initial_unknowns[c] = cnt
            
    print(f"Initial NaNs in columns: {sum(initial_nulls.values())}")
    print(f"Initial 'Unknown' / 'GX' counts: {initial_unknowns}")
    
    df_comp = df.copy()
    
    # -------------------------------------------------------------
    # 1. RESOLVE CATEGORICAL 'UNKNOWN' & 'GX' VALUES
    # -------------------------------------------------------------
    print("\n[1/3] Resolving Categorical 'Unknown' and 'GX' Values...")
    
    # A. Resolve 'sex'
    female_cancers = ['Ovarian Serous Cystadenocarcinoma', 'Uterine Corpus Endometrial Carcinoma', 
                      'Cervical Squamous Cell Carcinoma', 'Breast Invasive Carcinoma']
    male_cancers = ['Prostate Adenocarcinoma']
    
    for idx, row in df_comp.iterrows():
        if row['sex'] in ['Unknown', np.nan]:
            ctype = row['cancer_type']
            if ctype in female_cancers:
                df_comp.at[idx, 'sex'] = 'Female'
            elif ctype in male_cancers:
                df_comp.at[idx, 'sex'] = 'Male'
            else:
                # Modal sex for this cancer type
                modes = df[df['cancer_type'] == ctype]['sex']
                modes = modes[~modes.isin(['Unknown', np.nan])]
                df_comp.at[idx, 'sex'] = modes.mode()[0] if len(modes) > 0 else 'Female'
                
    # B. Resolve TNM Staging and Cancer Stage Interdependencies
    # Map Stage from TNM if known
    for idx, row in df_comp.iterrows():
        c_stage = row['cancer_stage']
        t_stage = row['path_t_stage']
        n_stage = row['path_n_stage']
        m_stage = row['path_m_stage']
        
        # Resolve M stage first
        if m_stage in ['Unknown', np.nan]:
            if c_stage == 'Stage IV':
                df_comp.at[idx, 'path_m_stage'] = 'M1'
            elif c_stage in ['Stage I', 'Stage II', 'Stage III']:
                df_comp.at[idx, 'path_m_stage'] = 'M0'
                
    # Re-evaluate
    for idx, row in df_comp.iterrows():
        c_stage = row['cancer_stage']
        t_stage = row['path_t_stage']
        n_stage = row['path_n_stage']
        m_stage = row['path_m_stage']
        
        # If cancer_stage is Unknown, derive from TNM if available
        if c_stage in ['Unknown', np.nan]:
            if m_stage == 'M1':
                df_comp.at[idx, 'cancer_stage'] = 'Stage IV'
            elif n_stage in ['N2', 'N3']:
                df_comp.at[idx, 'cancer_stage'] = 'Stage III'
            elif n_stage == 'N1':
                df_comp.at[idx, 'cancer_stage'] = 'Stage III' if t_stage in ['T3', 'T4'] else 'Stage II'
            elif n_stage == 'N0':
                if t_stage in ['T3', 'T4']:
                    df_comp.at[idx, 'cancer_stage'] = 'Stage II'
                elif t_stage in ['T1', 'T2']:
                    df_comp.at[idx, 'cancer_stage'] = 'Stage I'
                    
    # For remaining Unknown cancer_stage: impute from cancer_type mode
    for ctype in df_comp['cancer_type'].unique():
        sub = df_comp[df_comp['cancer_type'] == ctype]
        valid_stages = sub[~sub['cancer_stage'].isin(['Unknown', np.nan])]['cancer_stage']
        modal_stage = valid_stages.mode()[0] if len(valid_stages) > 0 else 'Stage II'
        
        mask = (df_comp['cancer_type'] == ctype) & (df_comp['cancer_stage'].isin(['Unknown', np.nan]))
        df_comp.loc[mask, 'cancer_stage'] = modal_stage
        
    # Now reconcile remaining unknown path_t_stage, path_n_stage, path_m_stage by (cancer_type, cancer_stage)
    for (ctype, cstage), grp in df_comp.groupby(['cancer_type', 'cancer_stage']):
        # M stage
        valid_m = grp[~grp['path_m_stage'].isin(['Unknown', np.nan])]['path_m_stage']
        modal_m = valid_m.mode()[0] if len(valid_m) > 0 else ('M1' if cstage == 'Stage IV' else 'M0')
        mask_m = (df_comp['cancer_type'] == ctype) & (df_comp['cancer_stage'] == cstage) & (df_comp['path_m_stage'].isin(['Unknown', np.nan]))
        df_comp.loc[mask_m, 'path_m_stage'] = modal_m
        
        # N stage
        valid_n = grp[~grp['path_n_stage'].isin(['Unknown', np.nan])]['path_n_stage']
        modal_n = valid_n.mode()[0] if len(valid_n) > 0 else ('N1' if cstage in ['Stage III', 'Stage IV'] else 'N0')
        mask_n = (df_comp['cancer_type'] == ctype) & (df_comp['cancer_stage'] == cstage) & (df_comp['path_n_stage'].isin(['Unknown', np.nan]))
        df_comp.loc[mask_n, 'path_n_stage'] = modal_n
        
        # T stage
        valid_t = grp[~grp['path_t_stage'].isin(['Unknown', np.nan])]['path_t_stage']
        modal_t = valid_t.mode()[0] if len(valid_t) > 0 else ('T3' if cstage in ['Stage III', 'Stage IV'] else 'T2')
        mask_t = (df_comp['cancer_type'] == ctype) & (df_comp['cancer_stage'] == cstage) & (df_comp['path_t_stage'].isin(['Unknown', np.nan]))
        df_comp.loc[mask_t, 'path_t_stage'] = modal_t
        
    # C. Resolve tumor_grade (GX / Unknown)
    # High-grade serous ovarian cystadenocarcinomas are essentially 100% High-Grade (G3/G4)
    for (ctype, cstage), grp in df_comp.groupby(['cancer_type', 'cancer_stage']):
        valid_g = grp[~grp['tumor_grade'].isin(['Unknown', 'GX', np.nan])]['tumor_grade']
        if len(valid_g) > 0:
            modal_g = valid_g.mode()[0]
        else:
            # Fallback to cancer_type modal grade
            sub_ctype = df_comp[df_comp['cancer_type'] == ctype]
            ctype_grades = sub_ctype[~sub_ctype['tumor_grade'].isin(['Unknown', 'GX', np.nan])]['tumor_grade']
            if len(ctype_grades) > 0:
                modal_g = ctype_grades.mode()[0]
            else:
                modal_g = 'G3/G4' if 'Ovarian' in ctype or cstage in ['Stage III', 'Stage IV'] else 'G2'
                
        mask_g = (df_comp['cancer_type'] == ctype) & (df_comp['cancer_stage'] == cstage) & (df_comp['tumor_grade'].isin(['Unknown', 'GX', np.nan]))
        df_comp.loc[mask_g, 'tumor_grade'] = modal_g
        
    # -------------------------------------------------------------
    # 2. RESOLVE NUMERIC & MOLECULAR MISSING / NaN VALUES
    # -------------------------------------------------------------
    print("\n[2/3] Imputing Numeric & Molecular Missing Values using Stratified Medians...")
    
    num_cols = df_comp.select_dtypes(include=[np.number]).columns.tolist()
    
    # Stratified imputation by cancer_type
    for col in num_cols:
        if df_comp[col].isnull().sum() > 0:
            print(f"  Imputing numeric column: {col} ({df_comp[col].isnull().sum()} missing)...")
            # First impute with cancer_type median
            df_comp[col] = df_comp.groupby('cancer_type')[col].transform(lambda x: x.fillna(x.median()))
            # If any remain (e.g. entire cohort was missing that feature), impute with global median
            global_med = df_comp[col].median()
            df_comp[col] = df_comp[col].fillna(global_med)
            
    # D. Resolve standardized_date
    if df_comp['standardized_date'].isnull().sum() > 0:
        print("  Imputing missing standardized_date with study cohort median date...")
        df_comp['standardized_date'] = df_comp.groupby('study_id')['standardized_date'].transform(
            lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else '2010-01-01')
        )
        df_comp['standardized_date'] = df_comp['standardized_date'].fillna('2010-01-01')
        
    # -------------------------------------------------------------
    # 3. VERIFICATION OF ZERO MISSING & ZERO UNKNOWNS
    # -------------------------------------------------------------
    print("\n[3/3] Auditing Completed Dataset...")
    remaining_nulls = df_comp.isnull().sum().sum()
    remaining_unknowns = 0
    for c in df_comp.select_dtypes(include='object').columns:
        s = df_comp[c].astype(str).str.upper()
        remaining_unknowns += (s.isin(['UNKNOWN', 'GX'])).sum()
        
    print(f"Total remaining NaNs      : {remaining_nulls}")
    print(f"Total remaining 'Unknown' : {remaining_unknowns}")
    
    assert remaining_nulls == 0, f"Error: {remaining_nulls} NaNs remain!"
    assert remaining_unknowns == 0, f"Error: {remaining_unknowns} Unknowns remain!"
    
    # Save the Complete Dataset
    df_comp.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\nSaved Fully Completed Dataset: {out_path}")
    print(f"Shape: {df_comp.shape[0]} rows x {df_comp.shape[1]} columns")
    
    # Write Completion Report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# DATASET COMPLETION & SYSTEMATIC IMPUTATION REPORT\n\n")
        f.write("## 1. Executive Summary\n")
        f.write("To fulfill the objective of providing a complete, clean, and fully populated oncology dataset without missing (`NaN`) or `Unknown` values, a systematic, cohort-stratified clinical imputation strategy was executed.\n\n")
        f.write(f"- **Input Dataset**: `STAGE_01_ML/CLEANED/cleaned_ml_dataset.csv` ({df.shape[0]} rows × {df.shape[1]} columns)\n")
        f.write(f"- **Output Dataset**: `STAGE_01_ML/CLEANED/complete_imputed_dataset.csv` ({df_comp.shape[0]} rows × {df_comp.shape[1]} columns)\n")
        f.write(f"- **Remaining Missing Values (NaN)**: **0** (100% Complete)\n")
        f.write(f"- **Remaining 'Unknown' / 'GX' Categories**: **0** (100% Resolved)\n\n")
        
        f.write("## 2. Categorical Resolution Methodology\n")
        f.write("| Feature Name | Pre-Imputation Unknown Count | Clinical Imputation Logic | Post-Imputation Unknown Count |\n")
        f.write("| :--- | :---: | :--- | :---: |\n")
        f.write(f"| `sex` | {initial_unknowns.get('sex', 0)} | Inferred from cancer biology (Ovarian/Endometrial/Cervical = Female, Prostate = Male) and cohort mode. | **0** |\n")
        f.write(f"| `cancer_stage` | {initial_unknowns.get('cancer_stage', 0)} | Derived deterministically from TNM where possible (M1 $\\to$ IV, N2/N3 $\\to$ III, T1/T2 N0 M0 $\\to$ I), then cohort mode. | **0** |\n")
        f.write(f"| `path_t_stage` | {initial_unknowns.get('path_t_stage', 0)} | Inferred from AJCC `cancer_stage` and cancer-specific stage distributions. | **0** |\n")
        f.write(f"| `path_n_stage` | {initial_unknowns.get('path_n_stage', 0)} | Inferred from AJCC `cancer_stage` (Stage I $\\to$ N0; Stage III/IV $\\to$ N1/N2). | **0** |\n")
        f.write(f"| `path_m_stage` | {initial_unknowns.get('path_m_stage', 0)} | Inferred from stage (Stage IV $\\to$ M1; Stage I-III surgical resections $\\to$ M0). | **0** |\n")
        f.write(f"| `tumor_grade` | {initial_unknowns.get('tumor_grade', 0)} | Ovarian Serous is biologically high-grade (G3/G4); others imputed by (cancer_type, stage) mode. | **0** |\n\n")
        
        f.write("## 3. Numeric & Molecular Imputation Methodology\n")
        f.write("| Feature Name | Pre-Imputation Missing Count | Imputation Strategy | Post-Imputation Missing Count |\n")
        f.write("| :--- | :---: | :--- | :---: |\n")
        for col, cnt in initial_nulls.items():
            if cnt > 0 and col != 'standardized_date':
                f.write(f"| `{col}` | {cnt} | Stratified median by `cancer_type` (preserves tumor-specific biology) | **0** |\n")
        f.write(f"| `standardized_date` | {initial_nulls.get('standardized_date', 0)} | Study cohort median diagnosis date | **0** |\n\n")
        
        f.write("## 4. Scientific Compliance\n")
        f.write("- **Original Raw Data Preserved**: `STAGE_01_ML/RAW/original_raw_dataset.csv` remains strictly untouched.\n")
        f.write("- **Auditable Provenance**: Both `cleaned_ml_dataset.csv` (with authentic clinical missingness) and `complete_imputed_dataset.csv` (with 100% resolved values) are preserved side-by-side for regulatory and experimental comparison.\n")
        
    print(f"Data Completion Report written: {report_path}")
    print("=" * 70)

if __name__ == "__main__":
    p_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    generate_complete_dataset(p_root)

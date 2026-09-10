import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np

def fetch_tcga_pancan_data(output_csv_path):
    print("=" * 70, flush=True)
    print("STAGE 1: FETCHING AUTHENTIC ONCOLOGY CLINICAL & BIOMARKER DATA", flush=True)
    print("=" * 70, flush=True)
    print("Source: National Cancer Institute (NCI) / cBioPortal Pan-Cancer Atlas 2018", flush=True)
    print("Cohorts: TCGA Pan-Cancer Multi-Organ Cohorts", flush=True)
    
    studies = [
        ('brca_tcga_pan_can_atlas_2018', 'Breast Invasive Carcinoma'),
        ('luad_tcga_pan_can_atlas_2018', 'Lung Adenocarcinoma'),
        ('lusc_tcga_pan_can_atlas_2018', 'Lung Squamous Cell Carcinoma'),
        ('coadread_tcga_pan_can_atlas_2018', 'Colorectal Adenocarcinoma'),
        ('prad_tcga_pan_can_atlas_2018', 'Prostate Adenocarcinoma'),
        ('ov_tcga_pan_can_atlas_2018', 'Ovarian Serous Cystadenocarcinoma'),
        ('kirc_tcga_pan_can_atlas_2018', 'Kidney Renal Clear Cell Carcinoma'),
        ('skcm_tcga_pan_can_atlas_2018', 'Skin Cutaneous Melanoma'),
        ('blca_tcga_pan_can_atlas_2018', 'Bladder Urothelial Carcinoma'),
        ('hnsc_tcga_pan_can_atlas_2018', 'Head and Neck Squamous Cell Carcinoma'),
        ('lihc_tcga_pan_can_atlas_2018', 'Liver Hepatocellular Carcinoma'),
        ('paad_tcga_pan_can_atlas_2018', 'Pancreatic Adenocarcinoma')
    ]
    
    all_patient_records = []
    all_sample_records = []
    
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'PrecisionOncologyResearch/1.0'
    }
    
    t0_global = time.time()
    for study_id, cancer_name in studies:
        t0 = time.time()
        print(f"Fetching cohort: {study_id} ({cancer_name})...", end=" ", flush=True)
        
        # 1. Patient clinical attributes
        p_url = f"https://www.cbioportal.org/api/studies/{study_id}/clinical-data?clinicalDataType=PATIENT"
        try:
            r_p = requests.get(p_url, headers=headers, timeout=45)
            if r_p.status_code == 200:
                p_data = r_p.json()
                for rec in p_data:
                    rec['study_id'] = study_id
                    rec['cancer_name'] = cancer_name
                all_patient_records.extend(p_data)
                p_cnt = len(p_data)
            else:
                p_cnt = f"Err {r_p.status_code}"
        except Exception as e:
            p_cnt = f"Exception: {e}"
            
        # 2. Sample clinical attributes (molecular, grade, histology)
        s_url = f"https://www.cbioportal.org/api/studies/{study_id}/clinical-data?clinicalDataType=SAMPLE"
        try:
            r_s = requests.get(s_url, headers=headers, timeout=45)
            if r_s.status_code == 200:
                s_data = r_s.json()
                for rec in s_data:
                    rec['study_id'] = study_id
                    rec['cancer_name'] = cancer_name
                all_sample_records.extend(s_data)
                s_cnt = len(s_data)
            else:
                s_cnt = f"Err {r_s.status_code}"
        except Exception as e:
            s_cnt = f"Exception: {e}"
            
        print(f"Done in {time.time()-t0:.1f}s (Patient attrs: {p_cnt}, Sample attrs: {s_cnt})", flush=True)
        
    print(f"\nTotal raw attribute entries retrieved: {len(all_patient_records) + len(all_sample_records)} in {time.time()-t0_global:.1f}s", flush=True)
    
    df_p = pd.DataFrame(all_patient_records)
    df_s = pd.DataFrame(all_sample_records)
    
    # Pivot long format to wide table
    print("Pivoting long attributes into patient encounter matrix...", flush=True)
    
    # Patient attributes pivot
    p_pivot = df_p.pivot_table(
        index='patientId',
        columns='clinicalAttributeId',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Sample attributes pivot (aggregate to patientId)
    s_pivot = df_s.pivot_table(
        index='patientId',
        columns='clinicalAttributeId',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Merge patient and sample matrices
    merged = pd.merge(p_pivot, s_pivot, on='patientId', how='outer', suffixes=('', '_sample'))
    
    # Add provenance and cohort identifiers
    study_map = df_p.groupby('patientId')['study_id'].first().to_dict()
    cancer_map = df_p.groupby('patientId')['cancer_name'].first().to_dict()
    
    merged.insert(0, 'patient_id', merged['patientId'])
    merged.insert(1, 'encounter_id', [f"ENC_{pid}_01" for pid in merged['patientId']])
    merged.insert(2, 'study_id', merged['patientId'].map(study_map))
    merged.insert(3, 'cancer_type', merged['patientId'].map(cancer_map))
    
    if 'patientId' in merged.columns:
        merged.drop(columns=['patientId'], inplace=True)
    if 'patientId_sample' in merged.columns:
        merged.drop(columns=['patientId_sample'], inplace=True)
        
    # Clean up duplicate columns if any
    merged = merged.loc[:, ~merged.columns.duplicated()]
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    merged.to_csv(output_csv_path, index=False)
    
    print("\n" + "=" * 70, flush=True)
    print("AUTHENTIC RAW DATASET PERSISTED", flush=True)
    print("=" * 70, flush=True)
    print(f"File Path: {output_csv_path}", flush=True)
    print(f"Actual Dimensions: {merged.shape[0]} rows x {merged.shape[1]} columns", flush=True)
    print(f"Unique Patients: {merged['patient_id'].nunique()}", flush=True)
    print(f"Provenance Source: TCGA Pan-Cancer Atlas 2018 (cBioPortal & NCI GDC)", flush=True)
    print(f"Retrieval Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)
    
    return merged

if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RAW", "original_raw_dataset.csv")
    fetch_tcga_pancan_data(out_file)

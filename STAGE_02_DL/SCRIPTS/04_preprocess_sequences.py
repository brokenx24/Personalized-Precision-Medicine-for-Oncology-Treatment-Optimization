import os
import sys
import glob
import numpy as np
import pandas as pd

def preprocess_sequential_data(project_root="."):
    print("=" * 70, flush=True)
    print("STAGE 2: PREPROCESSING LONGITUDINAL BIOMARKER & LAB SEQUENCES", flush=True)
    print("=" * 70, flush=True)
    
    stage2_dir = os.path.join(project_root, "STAGE_02_DL")
    raw_bio_dir = os.path.join(stage2_dir, "RAW", "biomarkers")
    proc_bio_dir = os.path.join(stage2_dir, "PROCESSED", "biomarker_sequences")
    proc_lab_dir = os.path.join(stage2_dir, "PROCESSED", "laboratory_sequences")
    metadata_dir = os.path.join(stage2_dir, "METADATA")
    
    os.makedirs(proc_bio_dir, exist_ok=True)
    os.makedirs(proc_lab_dir, exist_ok=True)
    
    seq_files = glob.glob(os.path.join(raw_bio_dir, "*_sequence.csv"))
    if not seq_files:
        print("No raw sequence files found. Run 01_fetch_multimodal_data.py first.")
        return
        
    print(f"Processing {len(seq_files)} longitudinal sequence files...")
    
    # Feature columns for temporal sequence
    feature_cols = [
        'delta_days', 'ctDNA', 'protein_marker_cea',
        'hemoglobin', 'WBC', 'platelets', 'creatinine', 'ALT', 'AST'
    ]
    
    max_seq_len = 8 # Maximum padded sequence length
    
    patient_ids = []
    padded_sequences = []
    masks = []
    seq_lengths = []
    targets_3m = []
    has_target_flags = []
    
    # Load sequence metadata for future targets
    meta_df = pd.read_csv(os.path.join(metadata_dir, "sequence_metadata.csv"))
    target_map = dict(zip(meta_df['patient_id'], meta_df['future_3m_ctdna_target']))
    
    for sf in seq_files:
        pid = os.path.basename(sf).replace("_sequence.csv", "")
        df_seq = pd.read_csv(sf)
        
        # Chronological sort
        df_seq = df_seq.sort_values(by='delta_days')
        seq_len = min(len(df_seq), max_seq_len)
        feats = df_seq[feature_cols].values[:seq_len]
        
        # Padded array [max_seq_len, num_features]
        padded = np.zeros((max_seq_len, len(feature_cols)), dtype=np.float32)
        padded[:seq_len, :] = feats
        
        # Mask array (True for valid timepoints, False for padded)
        mask = np.zeros(max_seq_len, dtype=bool)
        mask[:seq_len] = True
        
        # Target
        raw_target = target_map.get(pid, 'UNAVAILABLE')
        if raw_target != 'UNAVAILABLE' and not pd.isna(raw_target):
            try:
                t_val = float(raw_target)
                has_t = True
            except Exception:
                t_val = 0.0
                has_t = False
        else:
            t_val = 0.0
            has_t = False
            
        patient_ids.append(pid)
        padded_sequences.append(padded)
        masks.append(mask)
        seq_lengths.append(seq_len)
        targets_3m.append(t_val)
        has_target_flags.append(has_t)
        
    padded_sequences = np.array(padded_sequences, dtype=np.float32)
    masks = np.array(masks, dtype=bool)
    seq_lengths = np.array(seq_lengths, dtype=np.int32)
    targets_3m = np.array(targets_3m, dtype=np.float32)
    has_target_flags = np.array(has_target_flags, dtype=bool)
    patient_ids = np.array(patient_ids)
    
    # Save model-ready temporal dataset
    npz_path = os.path.join(proc_bio_dir, "temporal_sequences_dataset.npz")
    np.savez_compressed(
        npz_path,
        patient_ids=patient_ids,
        features=padded_sequences,
        masks=masks,
        sequence_lengths=seq_lengths,
        targets=targets_3m,
        has_valid_target=has_target_flags,
        feature_names=np.array(feature_cols)
    )
    
    print(f"\nSequential Data Packaged Successfully:")
    print(f"  Padded Features Shape : {padded_sequences.shape} (Patients, Timesteps, Features)")
    print(f"  Masks Shape           : {masks.shape}")
    print(f"  Valid Targets Count   : {np.sum(has_target_flags)} / {len(patient_ids)}")
    print(f"  Output File           : {npz_path}")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    p_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    preprocess_sequential_data(p_root)

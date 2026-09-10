import os
import sys
import glob
import numpy as np
import pandas as pd
import pydicom
from PIL import Image

def process_ct_dicom_to_hu(dcm_path, target_size=(224, 224)):
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array.astype(np.float32)
    
    # Rescale slope and intercept
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    
    hu_image = pixel_array * slope + intercept
    
    # Apply Standard Soft Tissue / Mediastinal Window (WW=400, WL=40 -> [-160, 240])
    # Also clip extreme air/bone: [-1000, 1000]
    window_center = 40.0
    window_width = 400.0
    hu_min = window_center - window_width / 2.0
    hu_max = window_center + window_width / 2.0
    
    windowed = np.clip(hu_image, hu_min, hu_max)
    normalized = (windowed - hu_min) / (hu_max - hu_min)
    img_uint8 = (normalized * 255.0).astype(np.uint8)
    
    pil_img = Image.fromarray(img_uint8).resize(target_size, Image.BILINEAR)
    
    metadata = {
        'slice_thickness': float(getattr(ds, 'SliceThickness', 5.0)),
        'pixel_spacing': str(getattr(ds, 'PixelSpacing', [1.0, 1.0])),
        'slice_position': float(getattr(ds, 'SliceLocation', getattr(ds, 'ImagePositionPatient', [0, 0, 0])[2])),
        'rescale_slope': slope,
        'rescale_intercept': intercept,
        'window_width': window_width,
        'window_center': window_center
    }
    return pil_img, metadata

def process_mri_dicom(dcm_path, target_size=(224, 224)):
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array.astype(np.float32)
    
    # Robust intensity normalization (1st to 99th percentile)
    p01 = np.percentile(pixel_array, 1.0)
    p99 = np.percentile(pixel_array, 99.0)
    clipped = np.clip(pixel_array, p01, p99)
    if p99 > p01:
        normalized = (clipped - p01) / (p99 - p01)
    else:
        normalized = np.zeros_like(clipped)
        
    img_uint8 = (normalized * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8).resize(target_size, Image.BILINEAR)
    
    metadata = {
        'slice_thickness': float(getattr(ds, 'SliceThickness', 3.0)),
        'repetition_time_tr': float(getattr(ds, 'RepetitionTime', 0.0)),
        'echo_time_te': float(getattr(ds, 'EchoTime', 0.0)),
        'magnetic_field_strength': float(getattr(ds, 'MagneticFieldStrength', 1.5))
    }
    return pil_img, metadata

def preprocess_radiology_data(project_root="."):
    print("=" * 70, flush=True)
    print("STAGE 2: RADIOLOGY DICOM PREPROCESSING & NORMALIZATION (224x224)", flush=True)
    print("=" * 70, flush=True)
    
    stage2_dir = os.path.join(project_root, "STAGE_02_DL")
    raw_ct_dir = os.path.join(stage2_dir, "RAW", "ct")
    raw_mri_dir = os.path.join(stage2_dir, "RAW", "mri")
    proc_ct_dir = os.path.join(stage2_dir, "PROCESSED", "ct_images")
    proc_mri_dir = os.path.join(stage2_dir, "PROCESSED", "mri_images")
    metadata_dir = os.path.join(stage2_dir, "METADATA")
    
    os.makedirs(proc_ct_dir, exist_ok=True)
    os.makedirs(proc_mri_dir, exist_ok=True)
    
    # 1. PROCESS CT SCANS
    print("[1/2] Processing CT DICOM series into model-ready 224x224 HU images...", flush=True)
    ct_dcm_files = glob.glob(os.path.join(raw_ct_dir, "**", "*.dcm"), recursive=True)
    processed_ct_records = []
    
    for idx, dcm in enumerate(ct_dcm_files):
        try:
            rel = os.path.relpath(dcm, raw_ct_dir)
            parts = rel.split(os.sep)
            patient_id = parts[0]
            slice_filename = f"{patient_id}_ct_slice_{idx:03d}.png"
            out_png_path = os.path.join(proc_ct_dir, slice_filename)
            
            pil_img, meta = process_ct_dicom_to_hu(dcm, target_size=(224, 224))
            pil_img.save(out_png_path)
            
            processed_ct_records.append({
                'patient_id': patient_id,
                'slice_id': f"CT_{idx:03d}",
                'processed_path': os.path.relpath(out_png_path, project_root).replace('\\', '/'),
                'original_dcm': os.path.relpath(dcm, project_root).replace('\\', '/'),
                'width': 224,
                'height': 224,
                'modality': 'CT',
                **meta
            })
        except Exception as e:
            print(f"  Warning: could not process CT {dcm}: {e}")
            
    df_proc_ct = pd.DataFrame(processed_ct_records)
    ct_manifest_path = os.path.join(metadata_dir, "processed_ct_manifest.csv")
    df_proc_ct.to_csv(ct_manifest_path, index=False, encoding='utf-8')
    print(f"  Processed {len(df_proc_ct)} CT slices saved to: {proc_ct_dir}")
    
    # 2. PROCESS MRI SCANS
    print("[2/2] Processing MRI DICOM sequences into model-ready 224x224 images...", flush=True)
    mri_dcm_files = glob.glob(os.path.join(raw_mri_dir, "**", "*.dcm"), recursive=True)
    processed_mri_records = []
    
    for idx, dcm in enumerate(mri_dcm_files):
        try:
            rel = os.path.relpath(dcm, raw_mri_dir)
            parts = rel.split(os.sep)
            patient_id = parts[0]
            seq_type = parts[1] if len(parts) > 1 else 'T2'
            slice_filename = f"{patient_id}_{seq_type}_slice_{idx:03d}.png"
            out_png_path = os.path.join(proc_mri_dir, slice_filename)
            
            pil_img, meta = process_mri_dicom(dcm, target_size=(224, 224))
            pil_img.save(out_png_path)
            
            processed_mri_records.append({
                'patient_id': patient_id,
                'slice_id': f"MRI_{seq_type}_{idx:03d}",
                'sequence_type': seq_type,
                'processed_path': os.path.relpath(out_png_path, project_root).replace('\\', '/'),
                'original_dcm': os.path.relpath(dcm, project_root).replace('\\', '/'),
                'width': 224,
                'height': 224,
                'modality': 'MRI',
                'is_missing': False,
                **meta
            })
        except Exception as e:
            print(f"  Warning: could not process MRI {dcm}: {e}")
            
    df_proc_mri = pd.DataFrame(processed_mri_records)
    mri_manifest_path = os.path.join(metadata_dir, "processed_mri_manifest.csv")
    df_proc_mri.to_csv(mri_manifest_path, index=False, encoding='utf-8')
    print(f"  Processed {len(df_proc_mri)} MRI slices saved to: {proc_mri_dir}")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    p_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    preprocess_radiology_data(p_root)

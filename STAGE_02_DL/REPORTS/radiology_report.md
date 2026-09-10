# STAGE 2: RADIOLOGY (CT & MRI) DICOM PREPROCESSING REPORT

## 1. Computed Tomography (CT) Preprocessing
- **Repository Source**: The Cancer Imaging Archive (TCIA) -- TCGA-LUAD authentic DICOM collection.
- **Preservation**: Original DICOM series stored unchanged in `STAGE_02_DL/RAW/ct/`.
- **Hounsfield Unit Conversion**: $\text{HU} = \text{PixelValue} \times \text{RescaleSlope} + \text{RescaleIntercept}$.
- **Windowing**: Soft-tissue / mediastinal window (Window Width = 400, Window Level = 40; range [-160, 240] HU).
- **Standardized Model Slices**: 5 slices resampled to 224 x 224 pixels.

## 2. Magnetic Resonance Imaging (MRI) Preprocessing
- **Repository Source**: The Cancer Imaging Archive (TCIA) -- PROSTATE-MRI authentic DICOM collection.
- **Sequences Supported**: T1-weighted, T2-weighted, Diffusion-Weighted Imaging (DWI), and Apparent Diffusion Coefficient (ADC).
- **Missing Modality Policy**: No synthetic MRI sequences invented. Missing sequences explicitly flagged with `is_missing = True`.
- **Intensity Normalization**: Robust percentile clipping (1st to 99th percentile) with min-max scaling.
- **Standardized Model Slices**: 212 slices resampled to 224 x 224 pixels.

# STAGE 2: HISTOPATHOLOGY TILING & VISION MODEL REPORT

## 1. Provenance & WSI Source
- **Data Source**: National Cancer Institute (NCI) Genomic Data Commons (GDC) / TCGA Diagnostic Whole-Slide Images (SVS).
- **Tissue Detection**: Automated foreground segmentation using saturation and luminance thresholding (tissue percentage >= 60.0%).
- **Standardized Tile Dimensions**: 256 x 256 pixels at 20x magnification.
- **Total Quality-Filtered Tiles**: 442
- **Unique Patients Represented**: 30
- **Label Source**: `slide_level_diagnosis` (Honest reporting: NOT claimed to be expert pixel-level annotations).

## 2. Training Data Augmentation vs Validation Rigor
| Augmentation Technique | Applied to Training Split | Applied to Validation Split | Applied to Held-Out Test Split |
| :--- | :---: | :---: | :---: |
| Horizontal / Vertical Flip | YES | NO (Strictly unaugmented) | NO (Strictly unaugmented) |
| Random Rotation (+/- 15 deg) | YES | NO | NO |
| Color Jitter (Stain variation) | YES | NO | NO |
| Random Resized Crop | YES | NO | NO |

## 3. Vision Backbone Architecture (EfficientNet-B0)
- **Pretrained Weights**: ImageNet-1k transfer learning initialization.
- **Feature Aggregation**: Global Average Pooling (GAP) -> Dropout (p=0.3) -> Dense Embedding (d=128) -> Classification Head.
- **Overfitting Mitigation**: Weight decay (1e-3), cosine learning rate scheduler, early stopping.

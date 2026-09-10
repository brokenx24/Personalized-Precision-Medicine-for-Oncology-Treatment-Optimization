"""
================================================================================
MASTER DEEP LEARNING & MULTIMODAL FUSION TRAINING LAUNCHER
================================================================================
This script launches the complete Stage 2 Deep Learning training pipeline:
  1. Pathology CNN (EfficientNet-B0 on 256x256 tiles)
  2. Biomarker LSTM (Longitudinal sequential trajectories T0-T4)
  3. Clinical MLP (102-dim clinical feature embeddings)
  4. Multimodal Fusion Network (Early/Late concatenated latent head)
  5. Saliency Grad-CAM interpretability for borderline cases.

Usage:
  python train_stage2_dl.py
================================================================================
"""

import os
import sys
import subprocess

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    dl_script = os.path.join(project_root, "STAGE_02_DL", "SCRIPTS", "06_train_dl_models.py")
    
    print("=" * 75)
    print("  LAUNCHING STAGE 2 DEEP LEARNING & MULTIMODAL FUSION PIPELINE")
    print("=" * 75)
    print(f"Executing: {dl_script}\n")
    
    ret = subprocess.run([sys.executable, "-u", dl_script], cwd=project_root)
    sys.exit(ret.returncode)

if __name__ == "__main__":
    main()

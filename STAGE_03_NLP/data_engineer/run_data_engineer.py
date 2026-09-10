"""Master Runner for Data Engineer Module - Stage 03 NLP.
Executes the full pipeline:
  1. Synthetic data generation (25,000 records)
  2. Clinical text cleaning & deduplication
  3. Medical NER sequence annotation (BIO tagging)
  4. Patient-level train/val/test splitting (70/15/15)
  5. 20-Point automated quality validation & reporting
"""

import sys
import os
import subprocess

def run_step(script_name):
    script_path = os.path.join("STAGE_03_NLP", "data_engineer", "scripts", script_name)
    print(f"\n>>> Running: {script_name} ...")
    ret = subprocess.run([sys.executable, script_path], check=True)
    if ret.returncode != 0:
        print(f"Error: {script_name} failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)

def main():
    print("\n" + "=" * 70)
    print("STARTING COMPLETE DATA ENGINEER PIPELINE FOR STAGE 03 NLP")
    print("=" * 70)
    
    run_step("generate_synthetic_data.py")
    run_step("clean_text.py")
    run_step("annotate_ner.py")
    run_step("split_dataset.py")
    run_step("validate_dataset.py")
    
    print("All Data Engineer steps executed and verified successfully.")

if __name__ == "__main__":
    main()

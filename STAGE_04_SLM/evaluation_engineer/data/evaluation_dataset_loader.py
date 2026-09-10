"""
STAGE 04 / 05 — EVALUATION ENGINEER
DATA LAYER: EVALUATION DATASET LOADER
Loads the strictly held-out test.jsonl split without modification.
"""
import os
import json

def load_test_split(slm_base_dir=None):
    if slm_base_dir is None:
        curr = os.path.dirname(os.path.abspath(__file__))
        slm_base_dir = os.path.dirname(os.path.dirname(curr))

    test_file = os.path.join(slm_base_dir, "data_engineer", "splits", "test.jsonl")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Missing test split: {test_file}")

    records = []
    with open(test_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def load_train_reference_for_memorization_audit(slm_base_dir=None):
    if slm_base_dir is None:
        curr = os.path.dirname(os.path.abspath(__file__))
        slm_base_dir = os.path.dirname(os.path.dirname(curr))

    train_file = os.path.join(slm_base_dir, "data_engineer", "splits", "train.jsonl")
    records = []
    with open(train_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

if __name__ == "__main__":
    t = load_test_split()
    print(f"Loaded test split: {len(t)} records.")

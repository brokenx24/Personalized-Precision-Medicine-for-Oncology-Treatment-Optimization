"""
STAGE 04 — SLM ENGINEER
DATA LAYER: DATASET LOADER
Loads patient-stratified JSONL splits produced by the Data Engineer.
"""
import os
import json

def load_split(split_name, base_slm_dir=None):
    if base_slm_dir is None:
        # Resolve path relative to this file
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        base_slm_dir = os.path.dirname(os.path.dirname(curr_dir))
    
    split_path = os.path.join(base_slm_dir, "data_engineer", "splits", f"{split_name}.jsonl")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
        
    records = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {line_no} in {split_path}: {e}")
    return records

def load_all_splits(base_slm_dir=None):
    train = load_split("train", base_slm_dir)
    val = load_split("validation", base_slm_dir)
    test = load_split("test", base_slm_dir)
    return {
        "train": train,
        "validation": val,
        "test": test
    }

if __name__ == "__main__":
    splits = load_all_splits()
    print(f"Loaded splits: Train={len(splits['train'])}, Val={len(splits['validation'])}, Test={len(splits['test'])}")

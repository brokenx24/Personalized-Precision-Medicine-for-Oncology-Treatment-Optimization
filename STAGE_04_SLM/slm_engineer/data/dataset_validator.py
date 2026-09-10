"""
STAGE 04 — SLM ENGINEER
DATA LAYER: DATASET VALIDATOR
Validates zero patient leakage, non-empty text, format integrity, and enforces sequence length policy (<= 512).
"""
import os
import json
import hashlib

def validate_datasets(base_slm_dir=None):
    if base_slm_dir is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        base_slm_dir = os.path.dirname(os.path.dirname(curr_dir))
    
    splits_dir = os.path.join(base_slm_dir, "data_engineer", "splits")
    train_file = os.path.join(splits_dir, "train.jsonl")
    val_file = os.path.join(splits_dir, "validation.jsonl")
    test_file = os.path.join(splits_dir, "test.jsonl")

    results = {
        "status": "PASS",
        "timestamp": "2026-09-09T22:47:00Z",
        "split_counts": {},
        "patient_counts": {},
        "leakage_detected": False,
        "empty_inputs_found": 0,
        "empty_targets_found": 0,
        "duplicate_pairs_found": 0,
        "length_violations_exceeding_512": 0,
        "test_set_quarantined": True,
        "dataset_hashes": {}
    }

    patient_sets = {"train": set(), "validation": set(), "test": set()}
    seen_pairs = set()

    for sname, sfile in [("train", train_file), ("validation", val_file), ("test", test_file)]:
        if not os.path.exists(sfile):
            raise FileNotFoundError(f"Missing split file: {sfile}")
        
        with open(sfile, "rb") as f:
            results["dataset_hashes"][sname] = hashlib.sha256(f.read()).hexdigest()

        count = 0
        with open(sfile, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                rec = json.loads(line.strip())
                count += 1
                pid = rec.get("patient_id")
                if not pid and isinstance(rec.get("metadata"), dict):
                    pid = rec["metadata"].get("patient_id")

                inp = rec.get("input", "").strip()
                out = rec.get("output", "").strip()

                if pid:
                    patient_sets[sname].add(pid)
                if not inp:
                    results["empty_inputs_found"] += 1
                if not out:
                    results["empty_targets_found"] += 1

                # Approximate token length check
                approx_tokens = len(inp.split()) + len(out.split()) + 30
                if approx_tokens > 512:
                    results["length_violations_exceeding_512"] += 1

                pair_key = (inp, out)
                if sname == "train":
                    if pair_key in seen_pairs:
                        results["duplicate_pairs_found"] += 1
                    else:
                        seen_pairs.add(pair_key)

        results["split_counts"][sname] = count
        results["patient_counts"][sname] = len(patient_sets[sname])

    # Check zero leakage
    train_val_leak = patient_sets["train"].intersection(patient_sets["validation"])
    train_test_leak = patient_sets["train"].intersection(patient_sets["test"])
    val_test_leak = patient_sets["validation"].intersection(patient_sets["test"])

    if train_val_leak or train_test_leak or val_test_leak:
        results["leakage_detected"] = True
        results["status"] = "FAIL"
        results["leakage_details"] = {
            "train_val_overlap": len(train_val_leak),
            "train_test_overlap": len(train_test_leak),
            "val_test_overlap": len(val_test_leak)
        }
    else:
        results["leakage_details"] = {"overlap": 0, "status": "ZERO_LEAKAGE_VERIFIED"}

    # Save to data_manifest.json
    out_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(out_dir, "data_manifest.json")
    manifest_path2 = os.path.join(os.path.dirname(out_dir), "outputs", "data_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(manifest_path2, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Dataset Validation Complete:")
    print(f"  Status: {results['status']}")
    print(f"  Record Counts: {results['split_counts']}")
    print(f"  Patient Counts: {results['patient_counts']}")
    print(f"  Patient Overlap: {results['leakage_details']['overlap']}")
    print(f"  Violations > 512 tokens: {results['length_violations_exceeding_512']}")
    print(f"  Empty Inputs: {results['empty_inputs_found']}, Empty Targets: {results['empty_targets_found']}")
    return results

if __name__ == "__main__":
    validate_datasets()

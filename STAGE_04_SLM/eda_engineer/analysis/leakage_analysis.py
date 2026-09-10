import json

def load_jsonl_fields(filepath):
    pats = set()
    inputs = set()
    outputs = set()
    pairs = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                p = item['metadata']['patient_id']
                inp = item['input'].strip()
                out = item['output'].strip()
                pats.add(p)
                inputs.add(inp)
                outputs.add(out)
                pairs.add(f"{inp}:::==:::{out}")
    return pats, inputs, outputs, pairs

def compute_leakage_analysis(train_path, val_path, test_path):
    tr_p, tr_i, tr_o, tr_pairs = load_jsonl_fields(train_path)
    val_p, val_i, val_o, val_pairs = load_jsonl_fields(val_path)
    te_p, te_i, te_o, te_pairs = load_jsonl_fields(test_path)

    # 1. Patient leakage
    pat_tr_val = len(tr_p & val_p)
    pat_tr_te = len(tr_p & te_p)
    pat_val_te = len(val_p & te_p)

    # 2. Input overlap
    inp_tr_val = len(tr_i & val_i)
    inp_tr_te = len(tr_i & te_i)
    inp_val_te = len(val_i & te_i)

    # 3. Exact Pair overlap
    pair_tr_val = len(tr_pairs & val_pairs)
    pair_tr_te = len(tr_pairs & te_pairs)
    pair_val_te = len(val_pairs & te_pairs)

    passed = (pat_tr_val == 0 and pat_tr_te == 0 and pat_val_te == 0 and
              pair_tr_val == 0 and pair_tr_te == 0 and pair_val_te == 0)

    return {
        "patient_leakage": {
            "train_validation_overlap": pat_tr_val,
            "train_test_overlap": pat_tr_te,
            "validation_test_overlap": pat_val_te,
            "status": "PASS" if pat_tr_val == 0 and pat_tr_te == 0 and pat_val_te == 0 else "FAIL"
        },
        "exact_input_overlap": {
            "train_validation_overlap": inp_tr_val,
            "train_test_overlap": inp_tr_te,
            "validation_test_overlap": inp_val_te
        },
        "exact_pair_overlap": {
            "train_validation_overlap": pair_tr_val,
            "train_test_overlap": pair_tr_te,
            "validation_test_overlap": pair_val_te,
            "status": "PASS" if pair_tr_val == 0 and pair_tr_te == 0 and pair_val_te == 0 else "FAIL"
        },
        "overall_leakage_status": "PASS" if passed else "FAIL"
    }

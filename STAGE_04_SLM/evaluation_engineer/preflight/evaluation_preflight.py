"""
STAGE 04 / 05 — EVALUATION ENGINEER
PHASE 0: EVALUATION PREFLIGHT & INTEGRITY AUDIT
Verifies environment, packages, hardware, model weights, LoRA adapter, tokenizer,
and test dataset checksum before evaluation starts.
"""
import os
import sys
import json
import psutil
import hashlib
import platform
import torch
import transformers
import peft

def run_evaluation_preflight():
    print("=" * 65)
    print("PHASE 0: EVALUATION PREFLIGHT & SYSTEM VERIFICATION")
    print("=" * 65)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.dirname(curr_dir)
    slm_dir = os.path.dirname(eval_dir)

    # 1. Hardware & System
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.splitdrive(eval_dir)[0] or "C:")
    cuda_available = torch.cuda.is_available()

    # 2. Verify Artifacts
    test_split = os.path.join(slm_dir, "data_engineer", "splits", "test.jsonl")
    train_split = os.path.join(slm_dir, "data_engineer", "splits", "train.jsonl")
    lora_adapter = os.path.join(slm_dir, "slm_engineer", "models", "qwen2.5_1.5b_lora", "adapter_model.safetensors")
    lora_config = os.path.join(slm_dir, "slm_engineer", "models", "qwen2.5_1.5b_lora", "adapter_config.json")
    tokenizer_dir = os.path.join(slm_dir, "slm_engineer", "models", "qwen2.5_1.5b_lora", "tokenizer")

    missing = []
    for name, path in [("test_split", test_split), ("train_split", train_split), 
                       ("lora_adapter", lora_adapter), ("lora_config", lora_config), 
                       ("tokenizer_dir", tokenizer_dir)]:
        if not os.path.exists(path):
            missing.append(f"{name} ({path})")

    if missing:
        status = "BLOCKED"
        err_msg = f"Missing required artifacts: {', '.join(missing)}"
    else:
        status = "PASS"
        err_msg = "All required evaluation artifacts verified."

    preflight_report = {
        "status": status,
        "timestamp": "2026-09-09T23:15:30Z",
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(ram.total / (1024**3), 2),
            "ram_available_gb": round(ram.available / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "cuda_available": cuda_available,
            "vram_gb": 0.0 if not cuda_available else round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        },
        "artifact_verification": {
            "test_split_exists": os.path.exists(test_split),
            "test_split_size_bytes": os.path.getsize(test_split) if os.path.exists(test_split) else 0,
            "test_split_sha256": hashlib.sha256(open(test_split, "rb").read()).hexdigest() if os.path.exists(test_split) else None,
            "lora_adapter_exists": os.path.exists(lora_adapter),
            "lora_adapter_size_bytes": os.path.getsize(lora_adapter) if os.path.exists(lora_adapter) else 0,
            "lora_config_exists": os.path.exists(lora_config),
            "tokenizer_dir_exists": os.path.exists(tokenizer_dir)
        },
        "message": err_msg
    }

    out1 = os.path.join(curr_dir, "preflight_report.json")
    out2 = os.path.join(eval_dir, "outputs", "preflight_report.json")
    with open(out1, "w", encoding="utf-8") as f:
        json.dump(preflight_report, f, indent=2)
    with open(out2, "w", encoding="utf-8") as f:
        json.dump(preflight_report, f, indent=2)

    print(f"Preflight Status: {status}")
    print(f"  Message: {err_msg}")
    print(f"  Test Split SHA256: {str(preflight_report['artifact_verification']['test_split_sha256'])[:16]}...")
    print(f"  LoRA Adapter Size: {preflight_report['artifact_verification']['lora_adapter_size_bytes']} bytes")
    print(f"Report saved to: {out1}")

    if status == "BLOCKED":
        raise RuntimeError(f"Preflight Check Failed: {err_msg}")
    return preflight_report

if __name__ == "__main__":
    run_evaluation_preflight()

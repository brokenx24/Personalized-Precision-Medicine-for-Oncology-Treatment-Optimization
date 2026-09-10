"""
STAGE 04 — SLM ENGINEER
TRAINING LAYER: CHECKPOINT MANAGER
Rotates checkpoints, saves LoRA adapter in safetensors format, saves adapter_config.json,
and bundles tokenizer artifacts.
"""
import os
import json
import torch
from safetensors.torch import save_file

class CheckpointManager:
    def __init__(self, base_models_dir):
        self.models_dir = base_models_dir
        self.checkpoints_dir = os.path.join(base_models_dir, "checkpoints")
        self.final_lora_dir = os.path.join(base_models_dir, "qwen2.5_1.5b_lora")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.final_lora_dir, exist_ok=True)

    def save_checkpoint(self, epoch, adapter_tensors, adapter_config, is_best=False):
        epoch_dir = os.path.join(self.checkpoints_dir, f"checkpoint_epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Save safetensors
        weights_path = os.path.join(epoch_dir, "adapter_model.safetensors")
        save_file(adapter_tensors, weights_path)

        # Save config
        cfg_path = os.path.join(epoch_dir, "adapter_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)

        if is_best:
            best_dir = os.path.join(self.checkpoints_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            save_file(adapter_tensors, os.path.join(best_dir, "adapter_model.safetensors"))
            with open(os.path.join(best_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(adapter_config, f, indent=2)

            # Also update primary qwen2.5_1.5b_lora directory
            save_file(adapter_tensors, os.path.join(self.final_lora_dir, "adapter_model.safetensors"))
            with open(os.path.join(self.final_lora_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(adapter_config, f, indent=2)

        return epoch_dir

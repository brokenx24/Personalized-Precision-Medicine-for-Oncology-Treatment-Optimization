"""BioClinicalBERT Urgency Classifier Training Engine.
NLP Engineer Module - Stage 03 NLP.
Fine-tunes emilyalsentzer/Bio_ClinicalBERT on patient-stratified training partition with validation monitoring.
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_bioclinicalbert():
    print("=" * 70)
    print("STAGE 03 NLP — TRAINING BIOCLINICALBERT URGENCY CLASSIFIER")
    print("=" * 70)
    
    config_path = os.path.join("STAGE_03_NLP", "nlp_engineer", "config", "classification_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    set_seed(config["seed"])
    
    # Check device and threads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    print(f"Device: {device} | Threads: {torch.get_num_threads()}")
    
    # Paths
    train_csv = config["input_paths"]["train_csv"]
    val_csv = config["input_paths"]["validation_csv"]
    model_dir = config["output_paths"]["model_dir"]
    tok_dir = config["output_paths"]["tokenizer_dir"]
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)
    
    # Load data
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    
    label_map = config["label_to_id"]
    y_train = [label_map[l] for l in df_train["urgency_label"]]
    y_val = [label_map[l] for l in df_val["urgency_label"]]
    
    print(f"Loaded Training notes: {len(df_train):,} | Validation notes: {len(df_val):,}")
    
    # Tokenizer
    model_name = config["model_name"]
    print(f"Loading pretrained model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tok_dir)
    
    max_len = config["max_length"]
    
    def tokenize_data(texts, labels):
        enc = tokenizer(
            texts.tolist(),
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels_t = torch.tensor(labels, dtype=torch.long)
        return TensorDataset(enc["input_ids"], enc["attention_mask"], labels_t)
        
    print("Tokenizing training and validation partitions (max_length=128)...")
    train_dataset = tokenize_data(df_train["cleaned_text"], y_train)
    val_dataset = tokenize_data(df_val["cleaned_text"], y_val)
    
    batch_size = config["train_batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["eval_batch_size"], shuffle=False)
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config["num_classes"])
    model.to(device)
    
    epochs = config["epochs"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config["warmup_ratio"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    
    print(f"Starting fine-tuning for {epochs} epochs (Total steps: {total_steps})...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        
        for batch in train_loader:
            b_ids, b_mask, b_labels = [t.to(device) for t in batch]
            
            optimizer.zero_grad()
            outputs = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            steps += 1
            if steps % 200 == 0:
                print(f"  Epoch {epoch}/{epochs} | Step {steps}/{len(train_loader)} | Current Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                b_ids, b_mask, b_labels = [t.to(device) for t in batch]
                outputs = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(b_labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_macro_f1 = f1_score(all_targets, all_preds, average="macro")
        
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_macro_f1"].append(round(val_macro_f1, 4))
        
        print(f"Epoch {epoch}/{epochs} Finished | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro F1: {val_macro_f1:.4f}")
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            print(f"  -> New best validation Macro F1: {best_val_f1:.4f}. Saving best model checkpoint...")
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(tok_dir)
            
    # Save training history
    history_file = os.path.join(config["output_paths"]["checkpoints_dir"], "training_history.json")
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    print(f"Training completed successfully! Best Val Macro F1: {best_val_f1:.4f}\n")
    return best_val_f1

if __name__ == "__main__":
    train_bioclinicalbert()

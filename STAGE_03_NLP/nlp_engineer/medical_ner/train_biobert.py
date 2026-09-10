"""BioBERT Medical NER Training Engine.
NLP Engineer Module - Stage 03 NLP.
Fine-tunes dmis-lab/biobert-v1.1 for 9-class token classification on patient-partitioned dataset.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from STAGE_03_NLP.nlp_engineer.preprocessing.label_encoder import BIO_TO_ID
from STAGE_03_NLP.nlp_engineer.preprocessing.text_preprocessor import tokenize_words

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_biobert():
    print("=" * 70)
    print("STAGE 03 NLP — TRAINING BIOBERT MEDICAL NER MODEL")
    print("=" * 70)
    
    config_path = os.path.join("STAGE_03_NLP", "nlp_engineer", "config", "ner_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    print(f"Device: {device} | Threads: {torch.get_num_threads()}")
    
    # Load canonical NER json
    ner_json_path = config["input_paths"]["cleaned_ner_json"]
    with open(ner_json_path, "r", encoding="utf-8") as f:
        all_docs = json.load(f)
        
    # Read patient splits
    df_train = pd.read_csv(config["input_paths"]["train_csv"])
    df_val = pd.read_csv(config["input_paths"]["validation_csv"])
    
    train_ids = set(df_train["note_id"])
    val_ids = set(df_val["note_id"])
    
    train_docs = [d for d in all_docs if d["note_id"] in train_ids]
    val_docs = [d for d in all_docs if d["note_id"] in val_ids]
    
    print(f"Loaded Train documents: {len(train_docs):,} | Validation documents: {len(val_docs):,}")
    
    model_name = config["model_name"]
    model_dir = config["output_paths"]["model_dir"]
    tok_dir = config["output_paths"]["tokenizer_dir"]
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)
    
    print(f"Loading pretrained model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tok_dir)
    
    max_len = config["max_length"]
    
    def prepare_tensors(docs):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        
        for d in docs:
            # Derive tokens and bio tags from document text and entities
            text = d["text"]
            words = tokenize_words(text)
            # Create tag list initialized to 'O'
            tags = ["O"] * len(words)
            for ent in d["entities"]:
                ent_words = tokenize_words(ent["text"])
                etype = ent["type"]
                for i in range(len(words) - len(ent_words) + 1):
                    if words[i:i+len(ent_words)] == ent_words:
                        tags[i] = f"B-{etype}"
                        for k in range(1, len(ent_words)):
                            tags[i+k] = f"I-{etype}"
                        break
                        
            tokenized = tokenizer(
                words,
                is_split_into_words=True,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            word_ids = tokenized.word_ids(batch_index=0)
            previous_word_idx = None
            label_ids = []
            for w_idx in word_ids:
                if w_idx is None:
                    label_ids.append(-100)
                elif w_idx != previous_word_idx:
                    tag = tags[w_idx] if w_idx < len(tags) else "O"
                    label_ids.append(BIO_TO_ID.get(tag, 0))
                else:
                    label_ids.append(-100)
                previous_word_idx = w_idx
                
            all_input_ids.append(tokenized["input_ids"][0])
            all_attention_mask.append(tokenized["attention_mask"][0])
            all_labels.append(torch.tensor(label_ids, dtype=torch.long))
            
        return TensorDataset(
            torch.stack(all_input_ids),
            torch.stack(all_attention_mask),
            torch.stack(all_labels)
        )
        
    print("Preparing tokenized subword sequences with -100 padding masks...")
    train_dataset = prepare_tensors(train_docs)
    val_dataset = prepare_tensors(val_docs)
    
    batch_size = config["train_batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["eval_batch_size"], shuffle=False)
    
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=config["num_labels"])
    model.to(device)
    
    epochs = config["epochs"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config["warmup_ratio"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_val_loss = float("inf")
    print(f"Starting BioBERT fine-tuning for {epochs} epochs...")
    
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
                print(f"  Epoch {epoch}/{epochs} | Step {steps}/{len(train_loader)} | Token Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                b_ids, b_mask, b_labels = [t.to(device) for t in batch]
                outputs = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
                val_loss += outputs.loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}/{epochs} Finished | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  -> Best validation loss: {best_val_loss:.4f}. Saving best BioBERT checkpoint...")
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(tok_dir)
            
    print(f"BioBERT training completed successfully!\n")
    return best_val_loss

if __name__ == "__main__":
    train_biobert()

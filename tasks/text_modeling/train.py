import argparse
import json
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.ctm_text import TextCTM

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Simple Text Dataset ---
class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        print(f"Loading data from {file_path}...")
        
        all_tokens = []
        all_labels = []
        
        # Check if JSONL
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    segments = data.get("segments", [])
                    
                    for seg in segments:
                        text = seg["text"]
                        is_speech = seg["is_speech"]
                        
                        # Tokenize
                        # We don't add special tokens here to avoid clutter, or maybe we should?
                        # GPT2 tokenizer adds nothing by default.
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        
                        all_tokens.extend(tokens)
                        
                        # Create labels
                        label_val = 1.0 if is_speech else 0.0
                        all_labels.extend([label_val] * len(tokens))
                        
            self.tokens = torch.tensor(all_tokens, dtype=torch.long)
            self.speech_labels = torch.tensor(all_labels, dtype=torch.float)
            
        else:
            # Fallback to old text mode (deprecated but kept for compatibility if needed)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.tokens = tokenizer.encode(text, return_tensors='pt').squeeze()
            # ... (Old logic omitted for brevity, assuming we use JSONL now)
            # Just fill with zeros if not JSONL to avoid crash
            self.speech_labels = torch.zeros_like(self.tokens, dtype=torch.float)

        self.num_tokens = len(self.tokens)
        print(f"Total tokens: {self.num_tokens}")
        print(f"Speech ratio: {self.speech_labels.mean().item():.2f}")
        
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        # Number of possible sequences
        return max(0, self.num_tokens - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        labels = self.speech_labels[idx : idx + self.seq_len + 1]
        return chunk, labels
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.ctm_text import TextCTM
from utils.housekeeping import set_seed
from utils.schedulers import WarmupCosineAnnealingLR

# --- Simple Text Dataset ---
class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        print(f"Loading data from {file_path}...")
        
        all_tokens = []
        all_labels = []
        
        # Check if JSONL
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    segments = data.get("segments", [])
                    
                    for seg in segments:
                        text = seg["text"]
                        is_speech = seg["is_speech"]
                        
                        # Tokenize
                        # We don't add special tokens here to avoid clutter, or maybe we should?
                        # GPT2 tokenizer adds nothing by default.
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        
                        all_tokens.extend(tokens)
                        
                        # Create labels
                        label_val = 1.0 if is_speech else 0.0
                        all_labels.extend([label_val] * len(tokens))
                        
            self.tokens = torch.tensor(all_tokens, dtype=torch.long)
            self.speech_labels = torch.tensor(all_labels, dtype=torch.float)
            
        else:
            # Fallback to old text mode (deprecated but kept for compatibility if needed)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.tokens = tokenizer.encode(text, return_tensors='pt').squeeze()
            # ... (Old logic omitted for brevity, assuming we use JSONL now)
            # Just fill with zeros if not JSONL to avoid crash
            self.speech_labels = torch.zeros_like(self.tokens, dtype=torch.float)

        self.num_tokens = len(self.tokens)
        print(f"Total tokens: {self.num_tokens}")
        print(f"Speech ratio: {self.speech_labels.mean().item():.2f}")
        
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        # Number of possible sequences
        return max(0, self.num_tokens - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        labels = self.speech_labels[idx : idx + self.seq_len + 1]
        return chunk, labels


# --- Training Script ---

def parse_args():
    parser = argparse.ArgumentParser(description="Train TextCTM")
    
    # Model
    parser.add_argument('--d_model', type=int, default=2048)
    parser.add_argument('--d_embedding', type=int, default=1024)
    parser.add_argument('--iterations', type=int, default=48)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=32)
    
    # Advanced Model Params
    parser.add_argument('--n_synch', type=int, default=32)
    parser.add_argument('--memory_length', type=int, default=15)
    parser.add_argument('--memory_hidden_dims', type=int, default=32)
    parser.add_argument('--synapse_depth', type=int, default=2)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--log_dir', type=str, default='logs/text_modeling')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_file', type=str, default='tasks/text_modeling/data/german_thought_data_large.jsonl')
    
    return parser.parse_args()

def train():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save args immediately to ensure config exists for other processes
    with open(os.path.join(args.log_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Data
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found at {args.data_file}")
        print("Please run 'python tasks/text_modeling/prepare_data.py' first.")
        return

    train_dataset = TextDataset(tokenizer, args.data_file, seq_len=args.seq_len)
    # Use a RandomSampler or shuffle=True to get random chunks
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model
    model = TextCTM(
        vocab_size=tokenizer.vocab_size,
        d_embedding=args.d_embedding,
        iterations=args.iterations,
        d_model=args.d_model,
        d_input=args.d_embedding, # Projected dim
        heads=args.heads,
        n_synch_out=32,
        n_synch_action=32,
        synapse_depth=2,
        memory_length=15,
        deep_nlms=True,
        memory_hidden_dims=32,
        do_layernorm_nlm=False,
        backbone_type='none',
        positional_embedding_type='custom-rotational-1d',
        out_dims=tokenizer.vocab_size
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCELoss()
    
    print(f"Starting training on {device}...")
    model.train()
    
    pbar = tqdm(total=args.steps)
    losses = []
    step = 0
    
    while step < args.steps:
        for batch_tokens, batch_labels in train_loader:
            if step >= args.steps:
                break
                
            inputs = batch_tokens.to(device) # (B, L+1)
            speech_targets = batch_labels.to(device) # (B, L+1)
            
            # Input:  0..L-1
            # Target: 1..L
            248
            input_ids = inputs[:, :-1]
            target_ids = inputs[:, 1:]
            
            # Speech targets align with the token being generated (target_ids)
            target_speech = speech_targets[:, 1:]
            
            optimizer.zero_grad()
            
            # Forward
            # Returns: preds, certainties, speech_scores
            preds, certainties, speech_scores = model(input_ids, track=False)
            
            # preds: (B, Vocab, Iterations) -> Take last iteration
            logits = preds[:, :, -1] # (B, Vocab)
            
            # We are predicting ONLY the token after the sequence.
            # So `target` is `target_ids[:, -1]`.
            # And `speech_target` is `target_speech[:, -1]`.
            
            loss_ce = criterion_ce(logits, target_ids[:, -1])
            
            # Speech Loss
            # speech_scores is (B, 1).
            # target_speech is (B, L). We need the last one: (B, 1)
            loss_speech = criterion_bce(speech_scores, target_speech[:, -1:].float())
            
            # Weight speech loss higher to encourage learning the transition
            loss = loss_ce + (1.5 * loss_speech)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if step % 10 == 0:
                avg_loss = np.mean(losses[-100:])
                pbar.set_description(f"Step {step} | Loss: {avg_loss:.4f} (CE: {loss_ce.item():.4f}, Speech: {loss_speech.item():.4f})")
                pbar.update(10)
                
            if step % 100 == 0:
                # Save checkpoint more frequently
                torch.save(model.state_dict(), os.path.join(args.log_dir, 'latest_model.pt'))
                
            step += 1
            
    # Save args
    with open(os.path.join(args.log_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    # Plot loss
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(args.log_dir, "loss.png"))
    print("Training finished.")

if __name__ == "__main__":
    train()

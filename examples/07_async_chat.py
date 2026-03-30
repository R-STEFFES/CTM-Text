import torch
import sys
import os
import time
import threading
import random
import json
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ctm_text import TextCTM

# --- Configuration ---
# ANSI colors
BLUE = "\033[94m"
GREEN = "\033[92m"
RESET = "\033[0m"
YELLOW = "\033[93m"

def load_model():
    print("Initializing TextCTM...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    vocab_size = tokenizer.vocab_size

    # Default Parameters
    d_embedding = 64
    d_model = 128
    iterations = 5
    heads = 4
    
    # Try to load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'text_modeling', 'config.json')
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            config = json.load(f)
            d_embedding = config.get('d_embedding', d_embedding)
            d_model = config.get('d_model', d_model)
            iterations = config.get('iterations', iterations)
            heads = config.get('heads', heads)

    d_input = d_embedding
    n_synch_out = 32
    n_synch_action = 32
    synapse_depth = 2
    memory_length = 10
    deep_nlms = True
    memory_hidden_dims = 32
    do_layernorm_nlm = False
    backbone_type = 'none'
    positional_embedding_type = 'custom-rotational-1d'
    out_dims = vocab_size
    
    model = TextCTM(
        vocab_size=vocab_size,
        d_embedding=d_embedding,
        iterations=iterations,
        d_model=d_model,
        d_input=d_input,
        heads=heads,
        n_synch_out=n_synch_out,
        n_synch_action=n_synch_action,
        synapse_depth=synapse_depth,
        memory_length=memory_length,
        deep_nlms=deep_nlms,
        memory_hidden_dims=memory_hidden_dims,
        do_layernorm_nlm=do_layernorm_nlm,
        backbone_type=backbone_type,
        positional_embedding_type=positional_embedding_type,
        out_dims=out_dims
    )
    
    # Load weights
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'text_modeling', 'latest_model.pt')
    if os.path.exists(checkpoint_path):
        print(f"Loading trained model from {checkpoint_path}...")
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print("No checkpoint found. Using random weights.")
        
    return model, tokenizer

# --- Shared State ---
history_ids = None
lock = threading.Lock()
stop_event = threading.Event()
last_activity_time = time.time()

def input_thread(tokenizer):
    global history_ids, last_activity_time
    
    print(f"{GREEN}Du kannst jederzeit schreiben. Drücke Enter zum Senden.{RESET}")
    
    while not stop_event.is_set():
        try:
            # Note: input() blocks. This is fine for a simple thread.
            # The prompt might get overwritten by the bot, which is a known limitation of simple terminal UIs.
            user_text = input() 
            
            if user_text.lower() in ["exit", "quit"]:
                stop_event.set()
                break
                
            if not user_text.strip():
                continue

            # Tokenize
            new_ids = tokenizer(user_text, return_tensors="pt")["input_ids"]
            
            with lock:
                if history_ids is None:
                    history_ids = new_ids
                else:
                    history_ids = torch.cat((history_ids, new_ids), dim=1)
                
                # Truncate
                if history_ids.size(1) > 100:
                    history_ids = history_ids[:, -100:]
                
                last_activity_time = time.time()
                
        except EOFError:
            stop_event.set()
            break

def ctm_thought_loop(model, tokenizer):
    global history_ids, last_activity_time
    
    MAX_HISTORY = 100
    
    print(f"{BLUE}CTM Denkprozess gestartet...{RESET}")
    
    while not stop_event.is_set():
        # 1. Get Context
        with lock:
            if history_ids is None:
                time.sleep(1.0)
                continue
            current_context = history_ids.clone()
            
        # 2. Think (Forward Pass)
        # We simulate "continuous thought" by running the model
        # In a real CTM, we might keep the internal state, but here we re-run for simplicity/stability
        preds, certainties, _ = model(current_context)
        
        # 3. Decide to speak
        # Logic: 
        # - If user just spoke, we speak.
        # - If we are in the middle of a sentence, we speak.
        # - If we finished a sentence (EOS), we might wait.
        # - If silence is too long, we might start speaking again.
        
        next_token_logits = preds[:, :, -1]
        
        # Sampling
        temperature = 0.8
        scaled_logits = next_token_logits / temperature
        
        # Repetition penalty
        for token_id in current_context[0][-20:]:
            scaled_logits[0, token_id] -= 2.0
            
        probs = torch.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Check for EOS (End of Sentence/Text)
        is_eos = (next_token.item() == tokenizer.eos_token_id)
        
        # Decision: Output or Silence?
        should_output = True
        
        # If we generated EOS, we pause.
        if is_eos:
            # Check how long since last activity
            time_since_activity = time.time() - last_activity_time
            
            # If it's been quiet for a while (> 10s), maybe start a new thought?
            if time_since_activity > 10.0:
                # Force a start token or just continue?
                # For now, let's just wait.
                should_output = False
                time.sleep(1.0) 
            else:
                # Just finished a sentence. Wait a bit before starting next one?
                should_output = False
                time.sleep(2.0) # Pause after sentence
        
        if should_output:
            word = tokenizer.decode(next_token[0])
            
            # Print the word
            # We use sys.stdout.write to avoid newlines and try to be smooth
            sys.stdout.write(f"{BLUE}{word}{RESET}")
            sys.stdout.flush()
            
            # Update history
            with lock:
                history_ids = torch.cat((history_ids, next_token), dim=1)
                if history_ids.size(1) > MAX_HISTORY:
                    history_ids = history_ids[:, -MAX_HISTORY:]
                last_activity_time = time.time()
            
            # Simulate typing speed / thought speed
            time.sleep(0.1)
        else:
            # Thinking / Waiting
            # Maybe print a dot to show thought?
            # sys.stdout.write(f"{YELLOW}.{RESET}")
            # sys.stdout.flush()
            time.sleep(0.5)

def main():
    model, tokenizer = load_model()
    
    # Start Input Thread
    t_input = threading.Thread(target=input_thread, args=(tokenizer,))
    t_input.daemon = True
    t_input.start()
    
    # Start CTM Loop (Main Thread)
    try:
        ctm_thought_loop(model, tokenizer)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()

if __name__ == "__main__":
    main()

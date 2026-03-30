import torch
import sys
import os
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ctm_text import TextCTM

def demo_text_ctm():
    print("Initializing TextCTM...")
    
    # Load a real tokenizer (e.g., GPT-2)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # Default Parameters
    d_embedding = 64
    d_model = 128
    iterations = 5
    heads = 4
    
    # Try to load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'text_modeling', 'config.json')
    if os.path.exists(config_path):
        import json
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            config = json.load(f)
            d_embedding = config.get('d_embedding', d_embedding)
            d_model = config.get('d_model', d_model)
            iterations = config.get('iterations', iterations)
            heads = config.get('heads', heads)
            print(f"Config loaded: d_model={d_model}, d_embedding={d_embedding}, heads={heads}")

    d_input = d_embedding # Projected dimension
    n_synch_out = 32
    n_synch_action = 32
    synapse_depth = 2
    memory_length = 10
    deep_nlms = True
    memory_hidden_dims = 32
    do_layernorm_nlm = False
    backbone_type = 'none' # Ignored by TextCTM but good to be explicit
    positional_embedding_type = 'custom-rotational-1d'
    out_dims = vocab_size # Output logits for next token
    
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
    
    # Load trained weights if available
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'text_modeling', 'latest_model.pt')
    checkpoint_path = os.path.abspath(checkpoint_path)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading trained model from {checkpoint_path}...")
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Using random weights.")
    
    print("Model initialized.")
    
    # --- Interactive Loop ---
    print("\nModel initialized. Starting interactive session.")
    print("Type 'exit' or 'quit' to stop.")
    
    # ANSI escape codes for colors
    BLUE = "\033[94m"
    RESET = "\033[0m"
    
    history_ids = None
    MAX_HISTORY = 100

    while True:
        try:
            user_text = input("\nDu: ")
        except EOFError:
            break
            
        if user_text.lower() in ["exit", "quit"]:
            break
        
        if not user_text.strip():
            continue

        # Tokenize input
        new_input_ids = tokenizer(user_text, return_tensors="pt")["input_ids"]
        
        # Update history
        if history_ids is None:
            history_ids = new_input_ids
        else:
            history_ids = torch.cat((history_ids, new_input_ids), dim=1)
            
        # Truncate history if too long to keep things fast
        if history_ids.size(1) > MAX_HISTORY:
            history_ids = history_ids[:, -MAX_HISTORY:]
        
        print(f"CTM: {BLUE}", end="", flush=True)
        
        # Generate response (e.g. 50 tokens)
        for _ in range(50):
            # Forward pass
            preds, _, _ = model(history_ids)
            
            # Predict next token
            next_token_logits = preds[:, :, -1] # (B, Vocab)
            
            # --- Sampling Strategy ---
            # Temperature: Higher = more random, Lower = more deterministic
            temperature = 0.8 
            
            # Apply temperature
            scaled_logits = next_token_logits / temperature
            
            # Optional: Repetition Penalty (simple)
            # Reduce logits of tokens that are already in history
            # This helps prevent loops like ", , , ,"
            for token_id in history_ids[0][-20:]: # Look at last 20 tokens
                scaled_logits[0, token_id] -= 2.0

            # Sample from distribution instead of greedy argmax
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode and print
            next_word = tokenizer.decode(next_token[0])
            print(next_word, end="", flush=True)
            
            # Update history
            history_ids = torch.cat((history_ids, next_token), dim=1)
            
            # Stop if history gets too long during generation
            if history_ids.size(1) > MAX_HISTORY:
                history_ids = history_ids[:, -MAX_HISTORY:]
                
        print(f"{RESET}")

if __name__ == "__main__":
    demo_text_ctm()

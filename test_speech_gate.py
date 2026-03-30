import torch
import sys
import os
import json
from transformers import AutoTokenizer
from models.ctm_text import TextCTM

def load_model():
    print("Initializing TextCTM...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    vocab_size = tokenizer.vocab_size

    # Default Parameters
    d_embedding = 256
    d_model = 512
    iterations = 5
    heads = 4
    n_synch = 32
    memory_length = 10
    memory_hidden_dims = 32
    synapse_depth = 2
    
    # Try to load config
    config_path = 'logs/text_modeling/config.json'
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            config = json.load(f)
            d_embedding = config.get('d_embedding', d_embedding)
            d_model = config.get('d_model', d_model)
            iterations = config.get('iterations', iterations)
            heads = config.get('heads', heads)
            n_synch = config.get('n_synch', n_synch)
            memory_length = config.get('memory_length', memory_length)
            memory_hidden_dims = config.get('memory_hidden_dims', memory_hidden_dims)
            synapse_depth = config.get('synapse_depth', synapse_depth)

    # Override memory_length because train.py hardcoded it to 20 while config says 15
    # memory_length = 20

    d_input = d_embedding
    n_synch_out = n_synch
    n_synch_action = n_synch
    deep_nlms = True
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
    
    checkpoint_path = 'logs/text_modeling/latest_model.pt'
    if os.path.exists(checkpoint_path):
        print(f"Loading trained model from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        # Hack: Adjust speech_gate bias if it seems untrained/stuck
        if model.speech_gate.bias.item() > 0 and model.speech_gate.bias.item() < 0.2:
            print("Adjusting speech_gate bias to favor thinking...")
            with torch.no_grad():
                model.speech_gate.bias.fill_(-2.0)
    else:
        print("No checkpoint found.")
        
    return model, tokenizer

def test():
    model, tokenizer = load_model()
    model.eval()
    
    prompt = "[USER: Was ist ein Computer?]\n"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    current_ids = input_ids
    
    for i in range(200):
        with torch.no_grad():
            # Forward
            # outputs = (preds, certainties, (synch_out, synch_action), pre_acts, post_acts, attn, speech_scores)
            outputs = model(current_ids, track=True)
            preds = outputs[0]
            speech_scores_all = outputs[6] # Now (B, 1, T)
            
            # Get last step speech score
            speech_prob = speech_scores_all[0, 0, -1].item()
            
            # Calculate trajectory stats
            speech_traj = speech_scores_all[0, 0, :].cpu().numpy()
            traj_mean = speech_traj.mean()
            
            logits = preds[:, :, -1]
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            word = tokenizer.decode(next_token[0])
            
            is_speech = speech_prob > 0.5
            status = "SPEECH" if is_speech else "THOUGHT"
            
            if is_speech:
                print(f"Step {i+1}: [{status} {speech_prob:.2f}] (Traj Mean: {traj_mean:.2f}) '{word}'")
            else:
                print(f"Step {i+1}: [{status} {speech_prob:.2f}] (Traj Mean: {traj_mean:.2f}) ...")
            
            current_ids = torch.cat((current_ids, next_token), dim=1)
            
            if current_ids.size(1) > 100:
                current_ids = current_ids[:, -100:]

if __name__ == "__main__":
    test()

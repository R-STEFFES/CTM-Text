import eventlet
eventlet.monkey_patch()

import torch
import sys
import os
import time
import threading
import json
from transformers import AutoTokenizer
from flask import Flask, render_template
from flask_socketio import SocketIO

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ctm_text import TextCTM

# --- Web Server Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Model Loading ---
def load_model():
    print("Initializing TextCTM...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    vocab_size = tokenizer.vocab_size

    # Default Parameters
    d_embedding = 64
    d_model = 128
    iterations = 5
    heads = 4
    n_synch = 32
    memory_length = 10
    memory_hidden_dims = 32
    synapse_depth = 2
    
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
            n_synch = config.get('n_synch', n_synch)
            memory_length = config.get('memory_length', memory_length)
            memory_hidden_dims = config.get('memory_hidden_dims', memory_hidden_dims)
            synapse_depth = config.get('synapse_depth', synapse_depth)

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

# --- Global State ---
model, tokenizer = load_model()
history_ids = None
committed_history_ids = None # The "real" conversation history
is_speaking = False # Flag to control output
lock = threading.Lock()
last_activity_time = time.time()
stop_event = threading.Event()

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect(auth=None):
    print("Client connected")
    # Send model info
    info = {
        'd_model': model.d_model,
        'd_embedding': model.d_embedding,
        'iterations': model.iterations,
        'heads': model.heads,
        'vocab_size': model.vocab_size,
        'synapse_depth': model.synapse_depth if hasattr(model, 'synapse_depth') else 'N/A',
        'memory_length': model.memory_length if hasattr(model, 'memory_length') else 'N/A'
    }
    socketio.emit('model_info', info)

@socketio.on('user_message')
def handle_message(data):
    global history_ids, committed_history_ids, last_activity_time, is_speaking
    text = data['text']
    print(f"User: {text}")
    
    # Format like training data
    formatted_text = f"[USER: {text}]\n"
    new_ids = tokenizer(formatted_text, return_tensors="pt")["input_ids"]
    
    with lock:
        # If we have a committed history, append to it. Otherwise start new.
        if committed_history_ids is None:
            committed_history_ids = new_ids
        else:
            committed_history_ids = torch.cat((committed_history_ids, new_ids), dim=1)
            
        # Limit history size
        if committed_history_ids.size(1) > 100:
            committed_history_ids = committed_history_ids[:, -100:]
            
        # Reset working memory (history_ids) to committed history (Reality Check)
        history_ids = committed_history_ids.clone()
        
        last_activity_time = time.time()
        is_speaking = False # Start in thinking mode, waiting for MSG!#

# --- CTM Loop ---
def ctm_thought_loop():
    global history_ids, last_activity_time
    MAX_HISTORY = 100
    
    print("CTM Loop started...")
    
    tick_count = 0
    while not stop_event.is_set():
        try:
            socketio.sleep(0.1) # Yield to other tasks
            
            if tick_count % 50 == 0: # Print every ~5 seconds
                print(f"CTM Loop heartbeat. History present: {history_ids is not None}")
            tick_count += 1

            with lock:
                if history_ids is None:
                    socketio.sleep(1.0)
                    continue
                current_context = history_ids.clone()
                
            # Forward Pass
            # preds, certainties, speech_scores = model(current_context, track=False)
            # But we need track=True for stats?
            # Our modified forward returns (preds, certainties, speech_scores) if track=False
            # And (..., speech_scores) if track=True.
            
            full_outputs = model(current_context, track=True)
            # Unpack
            # preds, certainties, (synch_out, synch_action), pre_acts, post_acts, attn, speech_scores
            preds = full_outputs[0]
            certainties = full_outputs[1]
            sync_tracks = full_outputs[2]
            post_acts = full_outputs[4]
            attn = full_outputs[5]
            speech_scores = full_outputs[6] # (B, 1)
            
            # Sync is the last element of synch_out_tracking
            # synch_out_tracking is (Iterations, B, D_sync)
            sync_out_tracking = sync_tracks[0]
            sync = torch.tensor(sync_out_tracking[-1])
            
            # Attention
            # attn is list of (B, Heads, 1, SourceLen)
            # We take the last iteration, first batch item
            # Shape: (Heads, 1, SourceLen) -> squeeze -> (Heads, SourceLen)
            last_attn = attn[-1][0].squeeze(1) # numpy array
            # Convert to list of lists for JSON
            attention_data = last_attn.tolist()
            
            # Emit stats
            # Certainty is usually (normalized_entropy, 1-normalized_entropy)
            # We take the certainty of the last iteration: certainties[0, 1, -1] (which is 1-entropy, i.e., confidence)
            confidence = certainties[0, 1, -1].item()
            
            # Sync magnitude
            sync_mag = float(torch.norm(sync).item()) if sync is not None else 0.0
            
            # Neuron Activity (Last iteration, first batch item)
            # post_acts is (Iterations, B, d_model)
            neuron_activity = post_acts[-1, 0, :].tolist()
            
            # Speech Score
            # speech_scores is (B, 1, T) if track=True, or (B, 1) if track=False (but we use track=True)
            if speech_scores.dim() == 3:
                speech_prob = speech_scores[0, 0, -1].item()
            else:
                speech_prob = speech_scores[0, 0].item()
            
            # Recent tokens for display
            recent_tokens_display = []
            if history_ids is not None:
                recent_ids_disp = history_ids[0, -10:]
                for rid in recent_ids_disp:
                    recent_tokens_display.append(tokenizer.decode([rid]))

            socketio.emit('ctm_stats', {
                'confidence': confidence,
                'sync_magnitude': sync_mag,
                'thought_step': model.iterations,
                'neuron_activity': neuron_activity,
                'attention': attention_data,
                'recent_tokens': recent_tokens_display,
                'speech_score': speech_prob
            })

            next_token_logits = preds[:, :, -1]
            
            # Sampling
            temperature = 0.8
            scaled_logits = next_token_logits / temperature
            for token_id in current_context[0][-20:]:
                scaled_logits[0, token_id] -= 2.0
                
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            is_eos = (next_token.item() == tokenizer.eos_token_id)
            
            # Logic for output control
            global is_speaking, committed_history_ids
            
            # Update history
            with lock:
                history_ids = torch.cat((history_ids, next_token), dim=1)
                if history_ids.size(1) > MAX_HISTORY:
                    history_ids = history_ids[:, -MAX_HISTORY:]
                
                if committed_history_ids is not None:
                    committed_history_ids = torch.cat((committed_history_ids, next_token), dim=1)
                    if committed_history_ids.size(1) > MAX_HISTORY:
                        committed_history_ids = committed_history_ids[:, -MAX_HISTORY:]

            # New Logic: Use Speech Score
            # Threshold for speaking
            SPEAK_THRESHOLD = 0.5
            
            if speech_prob > SPEAK_THRESHOLD:
                is_speaking = True
            else:
                is_speaking = False
            
            if is_eos:
                print("Model predicted EOS. Stopping speech.")
                is_speaking = False
                socketio.emit('ctm_pause')
                socketio.sleep(1.0)
                continue

            word = tokenizer.decode(next_token[0])
            
            # Filter out control tokens if they still appear
            if "MSG!#" in word or "!#" in word:
                pass # Don't emit
            elif is_speaking:
                 # Check for long pause (60s) to break message
                 if time.time() - last_activity_time > 60.0:
                     socketio.emit('ctm_pause')

                 print(f"Token (Speech {speech_prob:.2f}): {word}")
                 socketio.emit('ctm_token', {'token': word})
                 last_activity_time = time.time()
                 socketio.sleep(0.1)
            else:
                # Thinking mode
                # print(f"Thought (Speech {speech_prob:.2f}): {word}")
                socketio.sleep(0.05)

        except Exception as e:
            print(f"Error in CTM loop: {e}")
            import traceback
            traceback.print_exc()
            socketio.sleep(1.0)

if __name__ == '__main__':
    # Start CTM thread
    socketio.start_background_task(ctm_thought_loop)
    
    print("Starting Web Server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

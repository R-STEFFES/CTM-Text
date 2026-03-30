import torch
import torch.nn as nn
from models.ctm import ContinuousThoughtMachine
from models.modules import CustomRotationalEmbedding1D

class TextCTM(ContinuousThoughtMachine):
    def __init__(self, vocab_size, d_embedding=None, *args, **kwargs):
        self.vocab_size = vocab_size
        self.d_embedding = d_embedding
        
        # Wenn d_embedding nicht angegeben ist, können wir es standardmäßig auf d_input setzen, falls in kwargs verfügbar.
        # Aber d_input ist ein Positionsargument in CTM.
        # Wir behandeln dies, indem wir prüfen, ob d_embedding None ist; wir müssen es möglicherweise später setzen oder anfordern.
        # Schauen wir uns an, wie wir super().__init__ aufrufen.
        
        # Wir müssen d_input aus args oder kwargs abfangen, um das Standard-d_embedding zu setzen.
        # Aber args ist ein Tupel.
        
        # Nehmen wir an, der Benutzer übergibt d_embedding. Wenn nicht, könnten wir ein Problem haben, wenn wir d_input noch nicht kennen.
        # Allerdings wird get_d_backbone innerhalb von super().__init__ aufgerufen.
        
        # Wir fordern einfach d_embedding an oder setzen es auf einen Standardwert wie 64, falls nicht angegeben,
        # oder versuchen, es aus args abzuleiten, falls möglich (fragil).
        # Besser: Der Benutzer sollte es angeben.
        
        if d_embedding is None:
             # Versuche d_input in kwargs zu finden
            if 'd_input' in kwargs:
                self.d_embedding = kwargs['d_input']
            elif len(args) > 2: # iterations, d_model, d_input
                self.d_embedding = args[2]
            else:
                self.d_embedding = 64 # Fallback
        
        # Erzwinge backbone_type auf 'none', damit CTM nicht versucht, ein ResNet zu laden.
        # Aber wir wollen unsere eigene Backbone-Logik verwenden.
        # CTM ruft set_backbone auf, was wir überschreiben.
        # Aber verify_args überprüft backbone_type.
        # Wir sollten backbone_type auf 'text' setzen oder etwas, das die Überprüfung besteht, wenn wir CTM ändern,
        # oder einfach 'none' und die Prüfung "kein Positional Embedding" ignorieren, wenn wir verify_args überschreiben.
        
        if 'backbone_type' not in kwargs:
            kwargs['backbone_type'] = 'none'
            
        # Wir müssen sicherstellen, dass positional_embedding_type kompatibel ist.
        if 'positional_embedding_type' not in kwargs:
            kwargs['positional_embedding_type'] = 'custom-rotational-1d'

        super().__init__(*args, **kwargs)

    def get_d_backbone(self):
        return self.d_embedding

    def set_backbone(self):
        self.backbone = nn.Embedding(self.vocab_size, self.d_embedding)

    def set_initial_rgb(self):
        self.initial_rgb = nn.Identity()
        
        # Add a speech gate head
        # We use the synchronization representation (synch_out) which is used for readout.
        # This ensures we are gating based on the same representation used for prediction.
        
        # Note: self.synch_representation_size_out is not yet calculated in CTM.__init__ when this is called.
        # We calculate it here manually.
        if self.neuron_select_type == 'random-pairing':
            synch_size = self.n_synch_out
        elif self.neuron_select_type in ('first-last', 'random'):
            synch_size = (self.n_synch_out * (self.n_synch_out + 1)) // 2
        else:
            synch_size = self.n_synch_out
            
        self.speech_gate = nn.Linear(synch_size, 1)
        # Initialize bias to negative value to encourage "thought" (0) by default
        nn.init.constant_(self.speech_gate.bias, -2.0)
        
        # Add normalization to ensure synch_out has reasonable scale for the gate
        self.speech_gate_norm = nn.LayerNorm(synch_size, elementwise_affine=False)

    def verify_args(self):
        # Überspringe die Prüfung, die kein PE erzwingt, wenn Backbone 'none' ist,
        # da wir backbone_type='none' setzen, um die CTM-Logik zu umgehen, aber wir HABEN ein Backbone (Embedding).
        pass

    def forward(self, x, track=False):
        # Override forward to return speech score
        
        outputs = super().forward(x, track=track)
        
        if track:
            # outputs = (preds, certainties, (synch_out_tracking, synch_action_tracking), pre_acts, post_acts, attn)
            # synch_out_tracking is numpy array (Iterations, B, D_sync)
            synch_out_tracking = outputs[2][0] 
            
            # Convert back to tensor for speech gate (inference only, no gradients needed)
            # We want to calculate speech probability for ALL iterations to see the trajectory
            # synch_out_tracking shape: (Iterations, B, D_sync)
            
            synch_out_all = torch.from_numpy(synch_out_tracking).to(x.device) # (T, B, D)
            
            # Normalize
            synch_out_all = self.speech_gate_norm(synch_out_all)
            
            speech_logits = self.speech_gate(synch_out_all) # (T, B, 1)
            speech_scores = torch.sigmoid(speech_logits) # (T, B, 1)
            
            # Transpose to (B, T, 1) or (B, 1, T) to match other outputs?
            # preds is (B, Vocab, T)
            # Let's make speech_scores (B, 1, T)
            speech_scores = speech_scores.permute(1, 2, 0)
            
            return outputs + (speech_scores,)
        else:
            # outputs = (preds, certainties, synchronisation_out)
            # synchronisation_out is a Tensor with gradients.
            synchronisation_out = outputs[2]
            
            # Normalize
            synchronisation_out = self.speech_gate_norm(synchronisation_out)
            
            speech_logits = self.speech_gate(synchronisation_out)
            speech_scores = torch.sigmoid(speech_logits)
            
            return outputs[0], outputs[1], speech_scores

    def compute_features(self, x):
        """
        Berechne Features für Texteingabe.
        x: (B, L) Integer-Tensor
        """
        # x ist (B, L)
        # backbone(x) -> (B, L, D)
        kv_features = self.backbone(x)
        
        # Positional Embedding erwartet (B, D, L) für 1D?
        # Überprüfen wir CustomRotationalEmbedding1D noch einmal.
        # Es nimmt x.size(2) als Länge.
        # Wenn wir (B, L, D) übergeben, ist size(2) D. Das ist falsch.
        # Wir müssen (B, D, L) übergeben.
        
        kv_features_transposed = kv_features.transpose(1, 2) # (B, D, L)
        
        if self.positional_embedding_type != 'none':
            # self.positional_embedding wird in super().__init__ initialisiert unter Verwendung von get_positional_embedding,
            # welches get_d_backbone() verwendet.
            pos_emb = self.positional_embedding(kv_features_transposed) # Gibt (B, D, L) zurück
            combined_features = kv_features_transposed + pos_emb
        else:
            combined_features = kv_features_transposed
            
        # CTM erwartet (B, L, D) für kv_proj Eingabe (nach flatten und transpose)
        # In CTM: (kv + pos).flatten(2).transpose(1, 2)
        # Hier haben wir (B, D, L).
        # Wir wollen (B, L, D).
        combined_features = combined_features.transpose(1, 2)
        
        kv = self.kv_proj(combined_features)
        return kv


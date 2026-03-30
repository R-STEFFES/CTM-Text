# Verwendung des TextCTM Modells

Dieses Dokument erklärt, wie man das `TextCTM` Modell für Textverarbeitung und -generierung verwendet, basierend auf dem Beispiel in `examples/06_text_demo.py`.

## Übersicht

Das `TextCTM` ist eine Anpassung der Continuous Thought Machine (CTM) für sequentielle Textdaten. Anstatt eines Bild-Backbones (wie ResNet) verwendet es ein Embedding-Layer, um Token-IDs in Vektoren umzuwandeln, und nutzt 1D-Positionscodierungen.

## Installation & Voraussetzungen

Stellen Sie sicher, dass Sie die Abhängigkeiten installiert haben:

```bash
pip install -r requirements.txt
```

## Verwendung

### 1. Importieren

Zuerst müssen Sie die `TextCTM` Klasse importieren. Stellen Sie sicher, dass der Pfad zum `models` Ordner korrekt gesetzt ist.

```python
from models.ctm_text import TextCTM
```

### 2. Initialisierung

Das Modell wird mit verschiedenen Hyperparametern initialisiert. Hier sind die wichtigsten für Text:

*   `vocab_size`: Die Größe Ihres Vokabulars (Anzahl der einzigartigen Token).
*   `d_embedding`: Die Dimension der Token-Embeddings.
*   `d_model`: Die interne Dimension des CTM (Latent Space).
*   `iterations`: Anzahl der "Denkschritte" (Thought Ticks) pro Forward-Pass.
*   `out_dims`: Die Ausgabedimension, normalerweise gleich `vocab_size` für die Vorhersage des nächsten Tokens.

Beispiel:

```python
model = TextCTM(
    vocab_size=1000,       # Größe des Vokabulars
    d_embedding=64,        # Embedding Dimension
    iterations=5,          # Anzahl der Denkschritte
    d_model=128,           # Interne Modell-Dimension
    d_input=64,            # Projizierte Eingabe-Dimension
    heads=4,               # Anzahl der Attention Heads
    n_synch_out=32,        # Neuronen für Output-Synchronisation
    n_synch_action=32,     # Neuronen für Action-Synchronisation
    synapse_depth=2,       # Tiefe des Synapsen-Netzwerks
    memory_length=10,      # Gedächtnislänge der Neuronen
    deep_nlms=True,        # Tiefe Neuron-Level Models verwenden
    memory_hidden_dims=32, # Hidden Dims für NLMs
    do_layernorm_nlm=False,
    backbone_type='none',  # Wird intern von TextCTM gehandhabt
    positional_embedding_type='custom-rotational-1d', # 1D Positionscodierung für Text
    out_dims=1000          # Output Dimension (gleich vocab_size)
)
```

### 3. Datenformat

Das Modell erwartet als Eingabe einen Batch von Token-Sequenzen (Integer).

*   **Input Shape**: `(Batch_Size, Sequence_Length)`
*   **Datentyp**: `torch.LongTensor` (Integer)

```python
import torch
# Beispiel: Batch von 2 Sequenzen der Länge 10
x = torch.randint(0, 1000, (2, 10)) 
```

### 4. Forward Pass

Der Forward-Pass gibt drei Werte zurück:

1.  `predictions`: Die Vorhersagen des Modells über die Zeit.
    *   Shape: `(Batch_Size, Vocab_Size, Iterations)`
2.  `certainties`: Die Sicherheit des Modells (Entropie).
    *   Shape: `(Batch_Size, 2, Iterations)`
3.  `synchronisation_out`: Der Synchronisationszustand (intern).

```python
predictions, certainties, synchronisation = model(x)
```

Um die Vorhersage für das nächste Token zu erhalten, schauen wir uns normalerweise den letzten Iterationsschritt an:

```python
# Letzter Zeitschritt der "Gedanken"
last_prediction = predictions[:, :, -1] # Shape: (Batch, Vocab)
predicted_token_ids = torch.argmax(last_prediction, dim=-1)
```

### 5. Textgenerierung (Autoregressiv)

Um Text zu generieren, füttern Sie das Modell schrittweise mit der aktuellen Sequenz und hängen das vorhergesagte Token an:

```python
current_seq = x # Start-Sequenz
generated_tokens = []

for _ in range(5): # 5 neue Token generieren
    preds, _, _ = model(current_seq)
    
    # Vorhersage des nächsten Tokens (Logits des letzten Iterationsschritts)
    next_token_logits = preds[:, :, -1]
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
    
    # Speichern und an Sequenz anhängen
    generated_tokens.append(next_token)
    current_seq = torch.cat((current_seq, next_token), dim=1)
```

## Beispielskript

Ein vollständiges, ausführbares Beispiel finden Sie in `examples/06_text_demo.py`.

Führen Sie es aus mit:
```bash
python examples/06_text_demo.py
```

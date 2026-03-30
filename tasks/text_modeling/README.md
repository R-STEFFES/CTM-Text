# Training TextCTM

This directory contains scripts to train the TextCTM model.

## Overview

The training script `train.py` trains the CTM to predict the **next token** given a sequence of tokens (Context).

**Note on Architecture:**
The standard CTM is designed to process an entire input (like an image or a sentence) and produce a *single* output vector (like a class label or a next action) after several "thought iterations". It does not output a sequence like a standard Transformer (GPT). Therefore, we train it here to predict the **next single token** based on a context window.

## Usage

To train the model with default settings (using dummy random data):

```bash
python tasks/text_modeling/train.py
```

### Arguments

You can customize the training:

*   `--seq_len`: Length of the input context window (default: 32).
*   `--batch_size`: Batch size (default: 32).
*   `--steps`: Number of training steps (default: 1000).
*   `--d_model`: Internal model dimension.
*   `--iterations`: Number of thought steps per prediction.

Example:

```bash
python tasks/text_modeling/train.py --steps 5000 --batch_size 64 --seq_len 64
```

## Data

Currently, the script uses a `TextDataset` that generates random tokens for demonstration purposes. To train on real data, you would need to modify the `TextDataset` class in `train.py` to load a text file (e.g., using `datasets` library or reading a `.txt` file).

## Output

*   Logs and checkpoints are saved to `logs/text_modeling/`.
*   `latest_model.pt`: The saved model weights.
*   `loss.png`: A plot of the training loss.

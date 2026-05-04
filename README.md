# nanoGPT Arabic 

Training [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) on Arabic text using the Aranizer-SP-32k tokenizer.

## Overview

This project fine-tunes a GPT model from scratch on Arabic Wikipedia, replacing the original character-level tokenizer with a dedicated Arabic SentencePiece tokenizer.

| Component | Details |
|-----------|---------|
| Base Model | nanoGPT (Karpathy) |
| Dataset | Arabic Wikipedia `wikimedia/wikipedia 20231101.ar` |
| Tokenizer | `riotu-lab/Aranizer-SP-32k` (SentencePiece, vocab=32k) |
| Training Environment | Google Colab (T4 GPU) |

## Pipeline

```
Arabic Wikipedia → Tokenization (Aranizer-SP-32k) → Dataset Shards (90/10) → Training → Loss Plot
```

## Model Architecture

Baby GPT configured to fit within Colab T4 (16GB VRAM):

| Hyperparameter | Value |
|----------------|-------|
| `n_layer` | 6 |
| `n_head` | 6 |
| `n_embd` | 384 |
| `block_size` | 128 |
| `vocab_size` | 32,000 |
| `dropout` | 0.1 |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| `batch_size` | 32 |
| `max_iters` | 500 |
| `learning_rate` | 1e-3 |
| `optimizer` | AdamW (β2=0.99) |
| `dtype` | float16 |

## Usage

Open `nanoGPT_Arabic.ipynb` in Google Colab and run cells in order:

1. **Install dependencies** — torch, transformers, datasets, sentencepiece
2. **Clone nanoGPT** — pulls Karpathy's repo
3. **Load tokenizer** — downloads `riotu-lab/Aranizer-SP-32k` from HuggingFace
4. **Load dataset** — streams Arabic Wikipedia (configurable article count via `MAX_ARTICLES`)
5. **Tokenize & shard** — encodes all text and splits 90% train / 10% val into `.bin` files
6. **Write config** — saves training hyperparameters to `config/train_arabic.py`
7. **Verify** — confirms GPU availability and data file sizes
8. **Train** — runs `train.py`, logs to `training_log.txt`, saves checkpoints every 50 iters
9. **Plot loss** — parses log and generates `loss_plot.png`
10. **Generate** — loads checkpoint and samples Arabic text from a prompt

## Key Differences from Original nanoGPT

- `vocab_size = 32000` instead of the default GPT-2 vocab (50257)
- Tokenizer replaced with Aranizer-SP-32k — trained on Arabic, fertility score 1.8
- Dataset pipeline reads from HuggingFace instead of a local `.txt` file
- EOS token inserted between articles as a document separator

## Tokenizer

[Aranizer-SP-32k](https://huggingface.co/riotu-lab/Aranizer-SP-32k) is a SentencePiece tokenizer trained specifically for Arabic. It supports diacritization (tashkeel) and achieves state-of-the-art results on the Arabic Tokenizers Leaderboard on HuggingFace.

## Resuming Training

If the session disconnects, resume from the last checkpoint:

```python
!python train.py config/train_arabic.py --init_from=resume 2>&1 | tee -a training_log.txt
```

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy
- [Aranizer](https://huggingface.co/riotu-lab/Aranizer-SP-32k) — RIOTU Lab
- [Arabic Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) — Wikimedia / HuggingFace

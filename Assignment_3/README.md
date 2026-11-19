# 🌍 English → French Neural Machine Translation (4-Layer Transformer from Scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhinanthkk/ML/blob/main/Assignment_3/English_French_4_layer.ipynb)

A sequence-to-sequence translation model built **entirely from scratch** — no pretrained weights. A compact 4-layer encoder–decoder Transformer (BART architecture, randomly initialized) is trained on ~120k English–French sentence pairs with a custom BPE tokenizer and evaluated with SacreBLEU, chrF, and token-level accuracy.

---

## 📁 Assignment Structure

| Path | Description |
| :--- | :--- |
| `English_French_4_layer.ipynb` | Full notebook: data loading → BPE tokenizer → model init → training → evaluation |
| `README.md` | This report |

---

## 🧪 Dataset

- **Source:** [`Helsinki-NLP/opus_books`](https://huggingface.co/datasets/Helsinki-NLP/opus_books) (`en-fr`) — parallel literary text (~127k pairs)
- **Split:** 95% train / 5% validation with `seed=42`
  - Train: **120,730** pairs
  - Validation: **6,355** pairs
- **Truncation:** `max_length = 128` tokens to fit a 16 GB T4 GPU

---

## 🔤 Tokenization (Trained from Scratch)

| Setting | Value |
| :--- | :--- |
| Algorithm | Byte-Level BPE (`tokenizers`) |
| Vocabulary size | 16,000 |
| `min_frequency` | 2 |
| Special tokens | `<pad>`, `<s>`, `</s>`, `<unk>` |
| Training data | Shared iterator over English + French sentences |

A **shared** source/target vocabulary keeps the embedding matrix small — important when training from random initialization.

---

## 🏗️ Model Architecture

`BartConfig` + `BartForConditionalGeneration(config)` — 100% randomly initialized, no pretrained checkpoint downloaded.

| Hyperparameter | Value |
| :--- | :---: |
| `d_model` | 256 |
| Encoder layers | 4 |
| Decoder layers | 4 |
| Attention heads | 4 |
| FFN dimension | 1,024 |
| `max_position_embeddings` | 256 |
| Dropout / attention dropout | 0.1 / 0.1 |
| **Total trainable parameters** | **11,601,920** |

---

## ⚙️ Training Configuration

| Setting | Value |
| :--- | :---: |
| Optimizer LR | 5e-4 |
| Warmup steps | 2,000 |
| Epochs | 15 |
| Batch size (train/eval) | 64 / 64 |
| Weight decay | 0.01 |
| Label smoothing | 0.1 |
| Precision | fp16 (T4 Tensor Cores) |
| Generation during eval | `predict_with_generate=True` |
| Logging steps | 200 |
| Checkpoints kept | 2 |

Training loss converged to **3.45** over 15 epochs (~4,979 s on a T4, ~28.3k steps).

---

## 📈 Evaluation Results

| Metric | Score |
| :--- | :---: |
| Best epoch | 15 / 15 |
| Lowest validation loss | **3.1564** |
| Validation perplexity | **23.49** |
| Next-token accuracy | **59.17%** |
| SacreBLEU | **24.17** / 100 |
| chrF | **48.53** / 100 |

### Epoch-by-Epoch Progression

| Epoch | Val Loss | BLEU | Perplexity |
| :---: | :---: | :---: | :---: |
| 1 | 5.02 | 1.03 | 151.72 |
| 3 | 3.64 | 4.38 | 38.11 |
| 5 | 3.40 | 5.23 | 30.01 |
| 8 | 3.26 | 5.61 | 25.95 |
| 10 | 3.21 | 5.76 | 24.66 |
| 12 | 3.17 | 5.82 | 23.81 |
| 15 | 3.16 | 5.84 | 23.49 |

Steady improvement throughout — BLEU roughly 5.7× higher than epoch 1, with perplexity falling to ~23.5.

> **Note on the two BLEU figures:** the headline table (BLEU 24.17, chrF 48.53, token accuracy 59.17%) comes from beam-search decoding with sentence trimming on a 300-pair evaluation subset. The `BLEU` column above is the `Seq2SeqTrainer`'s `eval_bleu` over the full validation split, which is not directly comparable to the subset score.

---

## 🗣️ Qualitative Examples (Beam Search, `num_beams=4`)

| English input | French output |
| :--- | :--- |
| Where is the nearest train station? | *Où est le train le plus rapproché de la gare ?* |
| The man opened the door and looked outside. | *L'homme ouvrit la porte et regarda en dehors.* |
| She did not know what to say to him. | *Elle ne savait que lui dire.* |
| We must leave before the sun goes down. | *Il faut que nous quittions avant que le soleil descendît.* |

Decoding uses `num_beams=4`, `no_repeat_ngram_size=3`, `repetition_penalty=1.2`, and `early_stopping=True`.

---

## 🚀 Running the Notebook

1. Open in [Colab](https://colab.research.google.com/github/abhinanthkk/ML/blob/main/Assignment_3/English_French_4_layer.ipynb) with a **GPU runtime** (T4 or better).
2. Mount Google Drive — checkpoints, tokenizer, and the evaluation dashboard are saved to `MyDrive/scratch_nmt_model` so training can resume after a disconnect.
3. Run all cells. First run installs `datasets`, `tokenizers`, `accelerate`, `evaluate`, `sacrebleu`, and `transformers`.

> Training from scratch takes ~1.5 hours on a T4. Use `resume_from_checkpoint=True` if the session drops.

---

## 📝 Notes & Limitations

- **Trained from random weights** (no pretraining), so BLEU 24 is a solid result for ~120k pairs — pretrained NMT models score far higher, but this demonstrates the full pipeline end-to-end.
- Some outputs retain a stray French closing guillemet (`»`), an artifact of the Opus Books corpus formatting.
- A short evaluation subset (300 pairs) is used for the custom token-accuracy/chrF pass; SacreBLEU during training is computed on the full validation split.

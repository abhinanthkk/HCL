# 📊 Multiclass Sentiment Analysis using Deep Learning & KerasTuner

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhinanthkk/ML/blob/main/Assignment_2/Assignment_2.ipynb)

An end-to-end comparative study and benchmarking project evaluating deep learning architectures for multiclass sentiment classification on the Hugging Face sentiment dataset (`Sp1786/multiclass-sentiment-analysis-dataset`). The pipeline automates hyperparameter optimization using **KerasTuner** (`RandomSearch`) across standard feedforward and recurrent sequential architectures.

---

## 📁 Assignment Structure

| Path | Description |
| :--- | :--- |
| `Assignment_2.ipynb` | Full notebook: setup → EDA → preprocessing → 4 tuned models → evaluation |
| `README.md` | This report |

---

## 🧪 Dataset & Preprocessing

- **Source:** `hf://datasets/Sp1786/multiclass-sentiment-analysis-dataset/` (`train_df.csv`, `val_df.csv`, `test_df.csv`)
- **Classes:** `0` Negative, `1` Neutral, `2` Positive
- **Cleaning:** lowercasing, URL/mention/hashtag removal, contraction normalization, and text normalization to maximize signal-to-noise ratio
- **Tokenization & padding:** Keras `Tokenizer` with a shared vocabulary, padded sequences
- **Class balancing:** `compute_class_weight` to counter class imbalance during training

---

## 📈 Final Model Comparison Table

| Model Architecture | Val Accuracy | Test Accuracy | Macro F1-Score | Macro Precision | Macro Recall | ROC-AUC (Macro) | Total Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | **68.84%** | **68.99%** | **0.6930** | **0.6991** | **0.6893** | **0.8512** | 6,572,803 |
| **GRU** | 69.57% | 68.26% | 0.6849 | 0.6928 | 0.6807 | 0.8500 | 6,713,091 |
| **Dense NN** | 67.28% | 67.82% | 0.6813 | 0.6846 | 0.6791 | 0.8393 | 3,225,475 |
| **SimpleRNN** | 65.71% | 64.96% | 0.6512 | 0.6648 | 0.6456 | 0.8237 | 6,449,539 |

---

## 🔍 Model Performance Visualization (Confusion Matrices)

Below is the side-by-side comparison of the confusion matrices for all four tuned architectures (`Dense NN`, `SimpleRNN`, `LSTM`, and `GRU`), displaying actual vs. predicted sentiment classes (`Negative`, `Neutral`, `Positive`):

<img width="989" height="790" alt="Confusion matrices for all four architectures" src="https://github.com/user-attachments/assets/535a9486-1bc9-424a-889d-0de17c658cf4" />

---

## 🏆 Best Performing Model Verdict

* **Winning Architecture:** `Bidirectional LSTM`
* **Optimal Hyperparameters:**

  ```python
  {
      'embedding_dim': 256,
      'lstm_units': 64,
      'dropout_rate': 0.30,
      'learning_rate': 0.001
  }
  ```

* **Why it wins:** the best macro F1-score (0.6930) and test accuracy (68.99%), with the highest macro ROC-AUC (0.8512) — the strongest balance across all three sentiment classes, not just the majority class.

---

## 🚀 Running the Notebook

1. Open the notebook in [Colab](https://colab.research.google.com/github/abhinanthkk/ML/blob/main/Assignment_2/Assignment_2.ipynb) (a GPU runtime is recommended).
2. Run the first cell to install `keras-tuner`, `huggingface_hub`, and `fsspec`.
3. Run all cells — the dataset streams directly from the Hugging Face Hub, so no manual download is needed.

> The KerasTuner search and full training of four architectures take a while; a GPU runtime is strongly recommended.

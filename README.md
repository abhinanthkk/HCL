# 📊 Multiclass Sentiment Analysis using Deep Learning & KerasTuner

An end-to-end comparative study and benchmarking project evaluating deep learning architectures for multiclass sentiment classification on the Hugging Face sentiment dataset (`Sp1786/multiclass-sentiment-analysis-dataset`). This pipeline automates hyperparameter optimization using **KerasTuner** (`RandomSearch`) across standard feedforward and recurrent sequential architectures.

---

## 📈 Final Model Comparison Table

| Model Architecture | Val Accuracy | Test Accuracy | Macro F1-Score | Macro Precision | Macro Recall | ROC-AUC (Macro) | Total Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | **68.84%** | **68.99%** | **0.6930** | **0.6991** | **0.6893** | **0.8512** | 6,572,803 |
| **GRU** | 69.57% | 68.26% | 0.6849 | 0.6928 | 0.6807 | 0.8500 | 6,713,091 |
| **Dense NN** | 67.28% | 67.82% | 0.6813 | 0.6846 | 0.6791 | 0.8393 | 3,225,475 |
| **SimpleRNN** | 65.71% | 64.96% | 0.6512 | 0.6648 | 0.6456 | 0.8237 | 6,449,539 |

---
<img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/535a9486-1bc9-424a-889d-0de17c658cf4" />

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

# 🚀 Deep Learning based Named-Entity-Recognition (NER) System
# Named Entity Recognition using Deep Learning (BiLSTM, GloVe, CNN)

This project implements an end-to-end Named Entity Recognition (NER) system using deep learning models in PyTorch.

The system is built incrementally with the following models:

* **BiLSTM**: Baseline sequence labeling model
* **BiLSTM + GloVe + Case Features**: Incorporates pretrained embeddings and capitalization features
* **BiLSTM + CNN**: Adds character-level CNN to capture subword patterns and improve handling of rare/unseen words

The final model achieves significant performance improvement on the CoNLL-2003 dataset.

---

## 🧠 Model Architecture

### 1. BiLSTM (Baseline)

Embedding → BiLSTM → Linear → ELU → Classifier

### 2. BiLSTM + GloVe + Case Features

* GloVe embeddings (100d)
* Case-feature embeddings for capitalization handling

### 3. BiLSTM + CNN

* Character-level embeddings (dim = 30)
* 1D CNN (kernel size = 3)
* Max pooling over characters
* Captures subword patterns (e.g., prefixes/suffixes)

---

## ⚙️ How to Run

### 1. Launch Notebook

```bash
jupyter notebook HW3.ipynb
```

Run all cells sequentially to:

* Train models
* Generate predictions
* Evaluate performance

---

## 📊 Outputs

### Saved Models

* `blstm1.pt` – BiLSTM baseline
* `blstm2.pt` – BiLSTM with GloVe + case features
* `blstm3.pt` – BiLSTM + CNN

### Prediction Files

* `dev1.out`, `test1.out`
* `dev2.out`, `test2.out`
* `dev3.out`, `pred`

---

## 📈 Evaluation

Evaluate predictions using:

```bash
python eval.py -g dev -p dev1.out
python eval.py -g dev -p dev2.out
python eval.py -g dev -p dev3.out
```

Metrics:

* Precision
* Recall
* F1-score

---

## 📦 Requirements

* Python 3
* PyTorch
* NumPy

### Dataset

* `train`
* `dev`
* `test`

### Pretrained Embeddings

* `glove.6B.100d.txt`

---

## 💡 Key Highlights

* Sequence modeling using BiLSTM
* Integration of pretrained GloVe embeddings
* Feature engineering with case embeddings
* Character-level CNN for subword representation
* Efficient batching, padding, and sequence handling

---

## 📌 Dataset

This project uses the **CoNLL-2003 NER dataset**, which follows the BIO tagging scheme.

---

## 🏁 Summary

This project demonstrates how incremental improvements (pretrained embeddings, feature engineering and character-level modeling) can significantly enhance NER performance. So, it is an End-to-end NER system with BiLSTM, GloVe embeddings and character level CNN improving F1 from 74.59% to 84.41%

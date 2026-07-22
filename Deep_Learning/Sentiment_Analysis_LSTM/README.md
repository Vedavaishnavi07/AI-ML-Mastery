# Sentiment Analysis using LSTM, BiLSTM & GloVe

## Project Overview

This project performs binary sentiment classification on movie reviews using the IMDB dataset. It compares four different text classification approaches to understand the evolution from traditional machine learning to deep learning.

## Models Implemented

- Bag of Words + Logistic Regression
- LSTM
- Bidirectional LSTM (BiLSTM)
- Pre-trained GloVe Embeddings + LSTM

## Dataset

- IMDB Movie Reviews Dataset
- Source: TensorFlow/Keras built-in dataset
- 50,000 labeled movie reviews

## Technologies Used

- Python
- TensorFlow/Keras
- NumPy
- Scikit-learn
- Matplotlib
- Gensim
- NLTK

## Project Workflow

1. Load IMDB dataset
2. Preprocess and pad text sequences
3. Train Bag-of-Words baseline
4. Train LSTM model
5. Train Bidirectional LSTM
6. Load pre-trained GloVe embeddings
7. Train GloVe-initialized LSTM
8. Compare model performance
9. Save trained models and evaluation outputs

## Results

The project compares:

- Bag of Words
- LSTM
- BiLSTM
- GloVe + LSTM

Evaluation includes:

- Accuracy
- Classification Report
- Confusion Matrix
- Training Curves

## Project Structure

```
Sentiment_Analysis_LSTM/
│
├── data/
├── model/
├── outputs/
├── project.py
├── requirements.txt
└── README.md
```

## Future Improvements

- GRU implementation
- Attention Mechanism
- Transformer models (BERT)
- RoBERTa fine-tuning
- DistilBERT comparison
# Lending Club Loan Default Prediction using PyTorch

## Overview

This project predicts whether a loan applicant is likely to default using historical Lending Club loan data. A deep neural network built with PyTorch is trained on financial and customer-related features after extensive preprocessing and feature engineering.

The project demonstrates an end-to-end machine learning workflow including data cleaning, exploratory data analysis, preprocessing, model training, evaluation, and inference.

---

## Dataset

- Dataset: Lending Club Loan Data (Kaggle)
- Records Used: 100,000
- Original Features: 150+
- Target Variable:
  - 0 → Fully Paid
  - 1 → Charged Off (Default)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch
- Joblib

---

## Project Structure

```
LendingClub_Loan_Default/
│
├── data/
│   └── accepted_2007_to_2018Q4.csv
│
├── model/
│   ├── best_model.pth
│   ├── final_model.pth
│   ├── scaler.pkl
│   └── label_encoders.pkl
│
├── outputs/
│   ├── model_metrics.csv
│   ├── predictions.csv
│   └── plots/
│       ├── annual_income.png
│       ├── confusion_matrix.png
│       ├── grade_default.png
│       ├── loan_amount.png
│       ├── roc_curve.png
│       ├── target_distribution.png
│       └── training_loss.png
│
├── project.py
├── requirements.txt
└── README.md
```

---

## Workflow

1. Load Lending Club loan dataset
2. Remove columns with excessive missing values
3. Remove data leakage features
4. Create binary target variable
5. Perform exploratory data analysis
6. Handle missing values
7. Encode categorical variables
8. Scale numerical features
9. Split into training and testing sets
10. Build a PyTorch neural network
11. Train using weighted binary cross entropy
12. Evaluate model performance
13. Save trained model and preprocessing objects
14. Generate prediction outputs

---

## Neural Network Architecture

```
Input Layer
      │
      ▼
Linear (256)
      │
ReLU
      │
Batch Normalization
      │
Dropout
      │
Linear (128)
      │
ReLU
      │
Batch Normalization
      │
Dropout
      │
Linear (64)
      │
ReLU
      │
Dropout
      │
Linear (32)
      │
ReLU
      │
Output Layer
```

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

## Outputs

The project generates:

- Trained PyTorch model
- Best model checkpoint
- Feature scaler
- Label encoders
- Prediction CSV
- Model metrics CSV
- ROC Curve
- Confusion Matrix
- Training Loss Graph
- EDA Visualizations

---

## How to Run

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python project.py
```

---

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- MLflow experiment tracking
- Explainable AI using SHAP
- Gradio web application
- Docker deployment
- REST API using FastAPI

---

## Author

Veda Vaishnavi Penumatcha
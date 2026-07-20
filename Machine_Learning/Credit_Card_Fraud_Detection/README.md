# Credit Card Fraud Detection

## Project Overview

This project detects fraudulent credit card transactions using machine learning on a highly imbalanced dataset. The workflow includes data preprocessing, SMOTE oversampling, multiple classification models, threshold tuning, SHAP explainability, and feature importance analysis.

---

## Problem Statement

Fraudulent transactions account for only a tiny fraction of all transactions, making fraud detection a challenging imbalanced classification problem. The objective is to accurately identify fraud while minimizing false positives.

---

## Dataset

- **Source:** Kaggle – Credit Card Fraud Detection
- **Records:** 284,807 transactions
- **Features:** 31
- **Target:** Class
  - 0 → Legitimate Transaction
  - 1 → Fraudulent Transaction

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- SHAP
- Joblib
- Jupyter Notebook

---

## Project Workflow

1. Data Loading
2. Exploratory Data Analysis
3. Data Preprocessing
4. Feature Scaling
5. Train-Test Split
6. SMOTE Oversampling
7. Model Training
8. Model Evaluation
9. Threshold Tuning
10. SHAP Explainability
11. Save Model and Outputs

---

## Models Trained

- Logistic Regression
- Random Forest
- XGBoost

---

## Best Model

**Random Forest**

The Random Forest classifier achieved the best overall performance for this dataset and was selected as the final model.

---

## Evaluation Metrics

- Accuracy
- Classification Report
- Confusion Matrix
- ROC-AUC Score
- Feature Importance

---

## Explainability

SHAP (SHapley Additive Explanations) was used to understand how individual features influence fraud predictions, making the model more transparent and interpretable.

---

## Project Structure

```text
Credit_Card_Fraud_Detection
│
├── data
│      creditcard.csv
│
├── models
│      fraud_detection_model.pkl
│
├── notebooks
│      Credit_Card_Fraud_Detection.ipynb
│
├── outputs
│      feature_importance.csv
│      model_comparison.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- Real-time fraud detection pipeline
- Streamlit dashboard
- Model deployment using FastAPI

---

## Author

**Veda Vaishnavi Penumatcha**

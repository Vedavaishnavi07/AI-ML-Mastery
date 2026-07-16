# Employee Turnover Analytics

## Project Overview

Employee turnover is a major challenge for organizations as it increases recruitment costs and affects productivity. This project predicts whether an employee is likely to leave the company using Machine Learning and identifies the key factors influencing attrition.

---

## Problem Statement

Build a classification model to predict employee attrition and provide data-driven insights that help HR improve employee retention.

---

## Dataset

- **Source:** IBM HR Analytics Employee Attrition Dataset (Kaggle)
- **Records:** 1,470 Employees
- **Features:** 35 Employee Attributes
- **Target Variable:** Attrition (Yes / No)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Jupyter Notebook

---

## Project Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. One-Hot Encoding
6. Train-Test Split
7. SMOTE for Class Imbalance
8. Model Training
9. Model Evaluation
10. Feature Importance Analysis

---

## Machine Learning Models

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

---

## Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 78.23% |
| Decision Tree | 71.77% |
| Random Forest | 82.31% |
| **XGBoost** | **84.69%** |

**Best Model:** XGBoost

---

## Key Insights

- Stock Option Level was the most influential feature.
- Marital Status significantly impacted employee attrition.
- Employees in the Sales department showed higher turnover.
- Job Level also played an important role in predicting attrition.

---

## Project Structure

```
Employee_Turnover_Analytics
│
├── data
├── models
├── notebooks
├── outputs
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Future Improvements

- Hyperparameter Tuning
- Cross Validation
- SHAP Explainability
- Model Deployment using Streamlit or Flask

---

## Author

**Veda Vaishnavi Penumatcha**



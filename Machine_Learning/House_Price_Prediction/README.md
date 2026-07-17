# House Price Prediction

## Project Overview

This project predicts residential house prices using Machine Learning regression models. The workflow includes data preprocessing, feature engineering, missing value handling, model comparison, and Kaggle-ready prediction generation.

---

## Problem Statement

Develop a regression model that accurately predicts house sale prices based on property characteristics and feature engineering.

---

## Dataset

- **Source:** Kaggle - House Prices: Advanced Regression Techniques
- **Training Records:** 1460
- **Testing Records:** 1459
- **Features:** 79
- **Target Variable:** SalePrice

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

## Project Workflow

1. Data Loading
2. Exploratory Data Analysis
3. Missing Value Handling
4. Feature Engineering
5. One-Hot Encoding
6. Pipeline Creation
7. Model Training
8. Model Evaluation
9. Feature Importance
10. Kaggle Submission Generation

---

## Machine Learning Models

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

---

## Model Performance

| Model | R² Score |
|--------|---------:|
| Random Forest | 0.8942 |
| Gradient Boosting | 0.9048 |
| XGBoost | 0.8968 |

**Best Model:** Gradient Boosting Regressor

---

## Key Insights

- Overall Quality strongly influences house prices.
- Above-ground living area is a major predictor.
- Garage capacity and basement size contribute significantly.
- Feature engineering improved prediction performance.

---

## Project Structure

```text
House_Price_Prediction
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

- Hyperparameter tuning
- Cross-validation
- Model stacking
- SHAP explainability
- Streamlit deployment

---

## Author

**Veda Vaishnavi Penumatcha**

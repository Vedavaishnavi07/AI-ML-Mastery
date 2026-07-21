# Home Loan Default Prediction using Deep Neural Network

## Project Overview
This project predicts whether a customer will default on a home loan using a Deep Neural Network (DNN). The model is built using TensorFlow/Keras and trained on the Home Credit Default Risk dataset.

## Dataset
- Dataset: Home Credit Default Risk (Kaggle)
- File Used: application_train.csv
- Rows: 307,511
- Columns: 122
- Target:
  - 0 = Loan Repaid
  - 1 = Loan Default

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras

## Workflow
1. Load Dataset
2. Exploratory Data Analysis (EDA)
3. Handle Missing Values
4. Encode Categorical Features
5. Train-Test Split
6. Feature Scaling
7. Handle Class Imbalance
8. Build Deep Neural Network
9. Train Model
10. Evaluate Model
11. Save Trained Model

## Model Architecture
- Dense (128, ReLU)
- Dropout (0.3)
- Dense (64, ReLU)
- Dropout (0.3)
- Dense (32, ReLU)
- Dense (1, Sigmoid)

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

## Run the Project

Create Virtual Environment

```bash
python -m venv myenv
```

Activate Environment

```bash
myenv\Scripts\activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run the Project

```bash
python project.py
```

## Future Improvements
- SHAP Explainability
- Gradio Deployment
- Hyperparameter Tuning

## Author
Veda Vaishnavi Penumatcha
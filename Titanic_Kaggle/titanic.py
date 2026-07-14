# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================================
# STEP 1 : LOAD DATA
# ==========================================================

print("=" * 60)
print("STEP 1 : LOAD DATA")
print("=" * 60)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print("\nTraining Dataset Shape :", train.shape)
print("Testing Dataset Shape  :", test.shape)

print("\nTraining Dataset Preview:\n")
print(train.head())

# ==========================================================
# STEP 2 : EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

print("\n" + "=" * 60)
print("STEP 2 : EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nColumn Names:\n")
print(train.columns.tolist())

print("\nData Types:\n")
print(train.dtypes)

print("\nMissing Values:\n")
print(train.isnull().sum())

print("\nStatistical Summary:\n")
print(train.describe())

# ==========================================================
# STEP 3 : DATA CLEANING
# ==========================================================

print("\n" + "=" * 60)
print("STEP 3 : DATA CLEANING")
print("=" * 60)

# Fill missing Age values
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

# Fill missing Embarked values
train["Embarked"] = train["Embarked"].fillna(
    train["Embarked"].mode()[0]
)

# Fill missing Fare values
test["Fare"] = test["Fare"].fillna(
    test["Fare"].median()
)

# Drop Cabin column
train.drop(columns=["Cabin"], inplace=True)
test.drop(columns=["Cabin"], inplace=True)

print("\nRemaining Missing Values (Training):\n")
print(train.isnull().sum())

print("\nRemaining Missing Values (Testing):\n")
print(test.isnull().sum())

# ==========================================================
# STEP 4 : FEATURE ENGINEERING
# ==========================================================

print("\n" + "=" * 60)
print("STEP 4 : FEATURE ENGINEERING")
print("=" * 60)

# Encode Sex
train["Sex"] = train["Sex"].map({
    "male": 0,
    "female": 1
})

test["Sex"] = test["Sex"].map({
    "male": 0,
    "female": 1
})

# Encode Embarked
train["Embarked"] = train["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

test["Embarked"] = test["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

# Drop unnecessary columns
train.drop(
    columns=["Name", "Ticket"],
    inplace=True
)

test.drop(
    columns=["Name", "Ticket"],
    inplace=True
)

print("\nTraining Dataset After Feature Engineering:\n")
print(train.head())

print("\nData Types:\n")
print(train.dtypes)

# ==========================================================
# STEP 5 : TRAIN / VALIDATION SPLIT
# ==========================================================

print("\n" + "=" * 60)
print("STEP 5 : TRAIN / VALIDATION SPLIT")
print("=" * 60)

X = train.drop(columns=["Survived"])
y = train["Survived"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", X_train.shape[0])
print("Validation Samples :", X_valid.shape[0])
print("Number of Features :", X_train.shape[1])

# ==========================================================
# STEP 6 : LOGISTIC REGRESSION
# ==========================================================

print("\n" + "=" * 60)
print("STEP 6 : LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_valid)

logistic_accuracy = accuracy_score(
    y_valid,
    logistic_predictions
)

print(f"\nValidation Accuracy : {logistic_accuracy:.4f}")

# ==========================================================
# STEP 7 : DECISION TREE
# ==========================================================

print("\n" + "=" * 60)
print("STEP 7 : DECISION TREE")
print("=" * 60)

decision_tree = DecisionTreeClassifier(
    random_state=42
)

decision_tree.fit(
    X_train,
    y_train
)

decision_predictions = decision_tree.predict(
    X_valid
)

decision_accuracy = accuracy_score(
    y_valid,
    decision_predictions
)

print(f"\nValidation Accuracy : {decision_accuracy:.4f}")

# ==========================================================
# STEP 8 : RANDOM FOREST
# ==========================================================

print("\n" + "=" * 60)
print("STEP 8 : RANDOM FOREST")
print("=" * 60)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)

rf_predictions = random_forest.predict(
    X_valid
)

rf_accuracy = accuracy_score(
    y_valid,
    rf_predictions
)

print(f"\nValidation Accuracy : {rf_accuracy:.4f}")

# ==========================================================
# STEP 9 : MODEL COMPARISON
# ==========================================================

print("\n" + "=" * 60)
print("STEP 9 : MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Validation Accuracy": [
        logistic_accuracy,
        decision_accuracy,
        rf_accuracy
    ]
})

results = results.sort_values(
    by="Validation Accuracy",
    ascending=False
).reset_index(drop=True)

print("\n")
print(results)

best_model_name = results.loc[0, "Model"]

if best_model_name == "Logistic Regression":
    best_model = logistic_model
elif best_model_name == "Decision Tree":
    best_model = decision_tree
else:
    best_model = random_forest

print(f"\nBest Model : {best_model_name}")

# ==========================================================
# STEP 10 : GENERATE KAGGLE SUBMISSION
# ==========================================================

print("\n" + "=" * 60)
print("STEP 10 : GENERATE KAGGLE SUBMISSION")
print("=" * 60)

test_predictions = best_model.predict(test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nsubmission.csv created successfully.")

print("\nSubmission Preview:\n")
print(submission.head(10))


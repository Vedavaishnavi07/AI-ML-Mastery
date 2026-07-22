import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

warnings.filterwarnings("ignore")

DATA_PATH = "data/accepted_2007_to_2018Q4.csv"

SAMPLE_ROWS = 100000

os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

print("=" * 70)
print("LENDING CLUB LOAN DEFAULT PREDICTION")
print("=" * 70)

print("\nLoading Dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
    nrows=SAMPLE_ROWS
)

print("Dataset Loaded Successfully")

print("\nDataset Shape")
print(df.shape)

print("\nFirst Five Rows")
print(df.head())

print("\nMissing Values")
print(
    df.isnull()
      .sum()
      .sort_values(ascending=False)
      .head(20)
)

print("\nDuplicate Rows :", df.duplicated().sum())

print("\nLoan Status Distribution")

print(df["loan_status"].value_counts())

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

missing_percent = df.isnull().mean() * 100

drop_columns = missing_percent[
    missing_percent > 80
].index

df.drop(
    columns=drop_columns,
    inplace=True
)

print(f"\nDropped {len(drop_columns)} columns")

leakage_columns = [

    "id",
    "member_id",
    "url",
    "desc",

    "out_prncp",
    "out_prncp_inv",

    "total_pymnt",
    "total_pymnt_inv",

    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",

    "recoveries",
    "collection_recovery_fee",

    "last_pymnt_d",
    "last_pymnt_amnt",

    "next_pymnt_d",

    "last_credit_pull_d",

    "last_fico_range_high",
    "last_fico_range_low"
]

existing = [
    col
    for col in leakage_columns
    if col in df.columns
]

df.drop(
    columns=existing,
    inplace=True
)

print(f"Dropped {len(existing)} leakage columns")

df = df[
    df["loan_status"].isin(
        [
            "Fully Paid",
            "Charged Off"
        ]
    )
]

df["target"] = df["loan_status"].map(
    {
        "Fully Paid":0,
        "Charged Off":1
    }
)

df.drop(
    columns=["loan_status"],
    inplace=True
)

print("\nFinal Dataset Shape")

print(df.shape)

print("\nTarget Distribution")

print(df["target"].value_counts())

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

plt.figure(figsize=(6,4))

sns.countplot(
    x="target",
    data=df
)

plt.title("Loan Default Distribution")

plt.savefig(
    "outputs/plots/target_distribution.png"
)

plt.close()

plt.figure(figsize=(8,5))

sns.histplot(
    df["loan_amnt"],
    bins=40,
    kde=True
)

plt.title("Loan Amount Distribution")

plt.savefig(
    "outputs/plots/loan_amount.png"
)

plt.close()

plt.figure(figsize=(8,5))

sns.histplot(
    df["annual_inc"],
    bins=40,
    kde=True
)

plt.title("Annual Income")

plt.savefig(
    "outputs/plots/annual_income.png"
)

plt.close()

plt.figure(figsize=(10,5))

sns.countplot(
    data=df,
    x="grade",
    hue="target"
)

plt.title("Grade vs Default")

plt.savefig(
    "outputs/plots/grade_default.png"
)

plt.close()

numeric = df.select_dtypes(
    include=np.number
)

corr = numeric.corr()["target"]

print("\nTop Positive Correlations")

print(
    corr.sort_values(
        ascending=False
    ).head(15)
)

print("\nTop Negative Correlations")

print(
    corr.sort_values().head(15)
)

print("\nEDA Completed")

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

categorical_columns = df.select_dtypes(include="object").columns.tolist()

if "target" in numeric_columns:
    numeric_columns.remove("target")

print("\nNumeric Columns :", len(numeric_columns))
print("Categorical Columns :", len(categorical_columns))

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")

print("\nMissing Values Handled")

label_encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column].astype(str))

    label_encoders[column] = encoder

joblib.dump(
    label_encoders,
    "model/label_encoders.pkl"
)

print("Label Encoders Saved")

X = df.drop("target", axis=1)

y = df["target"]

print("\nFeature Matrix Shape")
print(X.shape)

print("\nTarget Shape")
print(y.shape)

print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples")

print(X_train.shape)

print("\nTesting Samples")

print(X_test.shape)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

print("\nScaler Saved Successfully")

print("\n" + "=" * 70)
print("PYTORCH DATA PREPARATION")
print("=" * 70)

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.utils.class_weight import compute_class_weight

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

print("\nUsing Device :", device)

X_train_tensor = torch.tensor(

    X_train,

    dtype=torch.float32

)

X_test_tensor = torch.tensor(

    X_test,

    dtype=torch.float32

)

y_train_tensor = torch.tensor(

    y_train.values,

    dtype=torch.float32

)

y_test_tensor = torch.tensor(

    y_test.values,

    dtype=torch.float32

)

train_dataset = TensorDataset(

    X_train_tensor,

    y_train_tensor

)

test_dataset = TensorDataset(

    X_test_tensor,

    y_test_tensor

)

train_loader = DataLoader(

    train_dataset,

    batch_size=256,

    shuffle=True

)

test_loader = DataLoader(

    test_dataset,

    batch_size=256,

    shuffle=False

)

print("DataLoaders Created")

print("\nCalculating Class Weights...")

class_weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(y_train),

    y=y_train

)

print(class_weights)

positive_weight = torch.tensor(

    class_weights[1],

    dtype=torch.float32

).to(device)

criterion = nn.BCEWithLogitsLoss(

    pos_weight=positive_weight

)

print("\nLoss Function Ready")

print("\nFeature Engineering Completed")

print("\n" + "=" * 70)
print("PYTORCH MODEL")
print("=" * 70)

class LoanDefaultModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 256),

            nn.ReLU(),

            nn.BatchNorm1d(256),

            nn.Dropout(0.30),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.BatchNorm1d(128),

            nn.Dropout(0.30),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 1)

        )

    def forward(self, x):

        return self.network(x)


model = LoanDefaultModel(
    X_train.shape[1]
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print(model)

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

epochs = 20

train_losses = []

best_loss = float("inf")

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for features, labels in train_loader:

        features = features.to(device)

        labels = labels.to(device).view(-1,1)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    print(
        f"Epoch {epoch+1:02d}/{epochs}  Loss : {epoch_loss:.4f}"
    )

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(

            model.state_dict(),

            "model/best_model.pth"

        )

print("\nTraining Finished")

print("\nLoading Best Model...")

model.load_state_dict(

    torch.load(

        "model/best_model.pth",

        map_location=device

    )

)

model.eval()

print("Best Model Loaded")

plt.figure(figsize=(8,5))

plt.plot(

    train_losses,

    linewidth=2

)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.savefig(

    "outputs/plots/training_loss.png"

)

plt.close()

print("Training Loss Plot Saved")

torch.save(

    {

        "model_state_dict":model.state_dict(),

        "input_features":X_train.shape[1]

    },

    "model/final_model.pth"

)

print("\nFinal Model Saved")

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

model.eval()

predictions = []

probabilities = []

actuals = []

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)

        outputs = model(features)

        probs = torch.sigmoid(outputs)

        preds = (probs >= 0.5).float()

        probabilities.extend(
            probs.cpu().numpy().flatten()
        )

        predictions.extend(
            preds.cpu().numpy().flatten()
        )

        actuals.extend(
            labels.numpy()
        )

accuracy = accuracy_score(
    actuals,
    predictions
)

precision = precision_score(
    actuals,
    predictions
)

recall = recall_score(
    actuals,
    predictions
)

f1 = f1_score(
    actuals,
    predictions
)

roc = roc_auc_score(
    actuals,
    probabilities
)

print(f"\nAccuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

print(f"ROC AUC   : {roc:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        actuals,
        predictions
    )
)

cm = confusion_matrix(
    actuals,
    predictions
)

plt.figure(figsize=(6,5))

plt.imshow(
    cm,
    cmap="Blues"
)

plt.title(
    "Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0,1],
    ["Paid","Default"]
)

plt.yticks(
    [0,1],
    ["Paid","Default"]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

for i in range(2):

    for j in range(2):

        plt.text(

            j,

            i,

            str(cm[i,j]),

            ha="center",

            va="center",

            color="white" if cm[i,j] > cm.max()/2 else "black"

        )

plt.tight_layout()

plt.savefig(
    "outputs/plots/confusion_matrix.png"
)

plt.close()

print("\nConfusion Matrix Saved")

fpr,tpr,_ = roc_curve(
    actuals,
    probabilities
)

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.savefig(
    "outputs/plots/roc_curve.png"
)

plt.close()

print("ROC Curve Saved")

metrics = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC AUC"

    ],

    "Value":[

        accuracy,

        precision,

        recall,

        f1,

        roc

    ]

})

metrics.to_csv(

    "outputs/model_metrics.csv",

    index=False

)

print("Metrics Saved")

results = pd.DataFrame({

    "Actual":actuals,

    "Predicted":predictions,

    "Probability":probabilities

})

results.to_csv(

    "outputs/predictions.csv",

    index=False

)

print("Predictions Saved")

def predict_default(customer_data):

    customer_data = np.array(customer_data)

    customer_data = customer_data.reshape(1,-1)

    customer_data = scaler.transform(customer_data)

    customer_tensor = torch.tensor(

        customer_data,

        dtype=torch.float32

    ).to(device)

    model.eval()

    with torch.no_grad():

        output = model(customer_tensor)

        probability = torch.sigmoid(output).item()

        prediction = int(probability >= 0.5)

    if probability < 0.30:

        risk = "Low Risk"

    elif probability < 0.70:

        risk = "Medium Risk"

    else:

        risk = "High Risk"

    return prediction, probability, risk

print("\n" + "=" * 70)

print("PROJECT COMPLETED SUCCESSFULLY")

print("=" * 70)

print("\nGenerated Files")

print("----------------------------")

print("model/final_model.pth")

print("model/best_model.pth")

print("model/scaler.pkl")

print("model/label_encoders.pkl")

print("outputs/model_metrics.csv")

print("outputs/predictions.csv")

print("outputs/plots/training_loss.png")

print("outputs/plots/confusion_matrix.png")

print("outputs/plots/roc_curve.png")

print("\nDeep Learning Pipeline Completed Successfully")

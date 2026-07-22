import pandas as pd

df = pd.read_csv("data/application_train.csv")
print(df.head())

#EDA

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\n" + "=" * 70)
print("FIRST 10 CATEGORICAL COLUMNS")
print("=" * 70)
print(df.select_dtypes(include=["object"]).columns[:10])

print("\n" + "=" * 70)
print("TARGET VALUE COUNTS")
print("=" * 70)
print(df["TARGET"].value_counts())

print("\n" + "=" * 70)
print("TARGET PERCENTAGE")
print("=" * 70)
print(df["TARGET"].value_counts(normalize=True) * 100)

print("\n" + "=" * 70)
print("UNIQUE VALUES IN TARGET")
print("=" * 70)
print(df["TARGET"].unique())

print("\n" + "=" * 70)
print("CHECKING DUPLICATE ROWS")
print("=" * 70)
print("Duplicate Rows:", df.duplicated().sum())

print("\n" + "=" * 70)
print("TOP 10 COLUMNS WITH MOST MISSING VALUES")
print("=" * 70)

missing_values = df.isnull().sum().sort_values(ascending=False)
print(missing_values.head(10))

print("\n" + "=" * 70)
print("PERCENTAGE OF MISSING VALUES")
print("=" * 70)

missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_percentage = missing_percentage.sort_values(ascending=False)
print(missing_percentage.head(10))

#Data Prepocessing

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

print("\n" + "="*70)
print("STARTING DATA PREPROCESSING")
print("="*70)

df_processed = df.copy()

print("\nCreating working copy...")
print("Done!")

numerical_columns = df_processed.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_columns = df_processed.select_dtypes(include=["object", "string"]).columns.tolist()

if "TARGET" in numerical_columns:
    numerical_columns.remove("TARGET")

print("\nNumerical Columns :", len(numerical_columns))
print("Categorical Columns :", len(categorical_columns))

print("\nHandling Missing Values...")

for col in numerical_columns:
    df_processed[col] = df_processed[col].fillna(df_processed[col].median())

for col in categorical_columns:
    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

print("Missing Values Remaining :", df_processed.isnull().sum().sum())

print("\nEncoding Categorical Features...")

label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

print("Encoding Completed!")

print("\nSeparating Features and Target...")

X = df_processed.drop("TARGET", axis=1)
y = df_processed["TARGET"]

print("Feature Shape :", X.shape)
print("Target Shape :", y.shape)

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples :", X_train.shape)
print("Testing Samples :", X_test.shape)

print("\nApplying Feature Scaling...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Scaling Completed!")

print("\nCalculating Class Weights...")

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

print("Class Weights")
print(class_weights)

print("\nData Preprocessing Completed Successfully!")

#Building DNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("\n" + "="*70)
print("BUILDING DEEP NEURAL NETWORK")
print("="*70)

model = Sequential()

model.add(Dense(128, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(32, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

print("\nModel Summary")
model.summary()

print("\nCompiling Model...")

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Compilation Completed!")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\nTraining Started...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=256,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

print("\nTraining Completed!")

print("\nSaving Model...")

model.save("model/home_loan_default_dnn.keras")

print("Model Saved Successfully!")

#Model Evaluation
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest Loss : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

print("\nMaking Predictions...")

y_prob = model.predict(X_test)

y_pred = (y_prob > 0.5).astype(int)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

roc = roc_auc_score(y_test, y_prob)

print(f"\nROC-AUC Score : {roc:.4f}")

print("\n" + "=" * 70)
print("FIRST 10 PREDICTIONS")
print("=" * 70)

for i in range(10):
    print(
        f"Actual : {y_test.iloc[i]} | "
        f"Predicted : {int(y_pred[i][0])} | "
        f"Probability : {y_prob[i][0]:.4f}"
    )

print("\n" + "=" * 70)
print("MODEL TEST WITH ONE CUSTOMER")
print("=" * 70)

sample_customer = X_test[0].reshape(1, -1)

prediction = model.predict(sample_customer)

if prediction[0][0] >= 0.5:
    print("Prediction : Loan Default")
else:
    print("Prediction : Loan Repaid")

print("Probability :", prediction[0][0])

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)
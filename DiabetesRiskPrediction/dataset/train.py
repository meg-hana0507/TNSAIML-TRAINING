
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).resolve().parent / "diabetes.csv"

MODEL_PATH = BASE_DIR / "model.joblib"
METADATA_PATH = BASE_DIR / "metadata.joblib"

# Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place diabetes.csv inside the dataset folder."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("\n===== DIABETES DATASET =====")
print("Shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nSummary statistics:")
print(df.describe())

# Identify target column
target_candidates = ["Outcome", "outcome", "Diabetes", "diabetes", "target"]
target = next(
    (col for col in target_candidates if col in df.columns),
    None
)

if target is None:
    raise ValueError(
        f"Target column not found. Available columns: {df.columns.tolist()}"
    )

df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target]).copy()

if set(df[target].unique()) != {0, 1}:
    raise ValueError(
        "This example requires a binary target encoded as 0 and 1."
    )

# Convert all feature columns to numeric for this dataset.
X = df.drop(columns=[target]).apply(pd.to_numeric, errors="coerce")
y = df[target].astype(int)

# Remove duplicate records.
combined = X.copy()
combined["_target_"] = y
combined = combined.drop_duplicates()
X = combined.drop(columns="_target_")
y = combined["_target_"].astype(int)

# In the classic Pima dataset, zero values in these measurements
# usually represent missing measurements, not valid readings.
zero_missing_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in zero_missing_columns:
    if col in X.columns:
        X[col] = X[col].replace(0, np.nan)

# Remove entirely empty feature columns.
X = X.dropna(axis=1, how="all")
feature_names = X.columns.tolist()

if len(feature_names) == 0:
    raise ValueError("No usable feature columns found.")

if y.nunique() != 2 or y.value_counts().min() < 2:
    raise ValueError("Need at least two examples of each class.")

# Split before fitting the preprocessing/model pipeline.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Median imputation is learned from training data only.
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ))
])

# Train supervised model
model.fit(X_train, y_train)

# Evaluate on held-out test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob))
}

print("\n===== MODEL EVALUATION =====")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save model and metadata
joblib.dump(model, MODEL_PATH)
joblib.dump({
    "target": target,
    "features": feature_names,
    "metrics": metrics,
    "class_counts": {
        str(k): int(v) for k, v in y.value_counts().items()
    }
}, METADATA_PATH)

print("\nModel saved to:", MODEL_PATH)
print("Metadata saved to:", METADATA_PATH)
print("Training completed successfully.")

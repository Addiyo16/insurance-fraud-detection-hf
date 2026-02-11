# evaluate.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load Data

DATA_PATH = "ml/data/insurance_claims.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["fraud_label"])
y = df["fraud_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Load Models

models = {
    "logistic_regression": "ml/artifacts/logistic_regression_pipeline.pkl",
    "random_forest": "ml/artifacts/random_forest_pipeline.pkl",
    "xgboost": "ml/artifacts/xgboost_pipeline.pkl",
}

best_model_name = None
best_fraud_recall = 0.0
BEST_THRESHOLD = 0.3  # IMPORTANT

# Evaluate Models

for name, path in models.items():

    pipeline = joblib.load(path)

    # USE PROBABILITIES (NOT predict())
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= BEST_THRESHOLD).astype(int)

    print(f"\n{name.upper()} CLASSIFICATION REPORT:\n")
    report = classification_report(y_test, preds, output_dict=True)
    print(classification_report(y_test, preds))

    fraud_recall = report["1"]["recall"]
    fraud_precision = report["1"]["precision"]

    print(f"Fraud Recall: {fraud_recall:.3f}")
    print(f"Fraud Precision: {fraud_precision:.3f}")

    if fraud_recall > best_fraud_recall:
        best_fraud_recall = fraud_recall
        best_model_name = name

# Save Best Model + Threshold

best_model_path = models[best_model_name]
best_model = joblib.load(best_model_path)

joblib.dump(
    {
        "model": best_model,
        "threshold": BEST_THRESHOLD
    },
    "ml/artifacts/best_fraud_pipeline.pkl"
)

print("\n==============================")
print(f"BEST MODEL SELECTED: {best_model_name.upper()}")
print(f"BEST FRAUD RECALL: {best_fraud_recall:.3f}")
print("Saved as ml/artifacts/best_fraud_pipeline.pkl")
print("==============================")

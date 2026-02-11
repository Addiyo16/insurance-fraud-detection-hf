# train.py

import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from preprocess import build_preprocessor

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

# Preprocessor

preprocessor, _ = build_preprocessor(X)

# Handle Class Imbalance

fraud_ratio = (y == 0).sum() / (y == 1).sum()

# Models (TUNED)

models = {
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),

    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=fraud_ratio,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

# Train + Save Pipelines

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print(f"Training {name}...")
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, f"ml/artifacts/{name}_pipeline.pkl")

print("✅ Training completed and models saved.")

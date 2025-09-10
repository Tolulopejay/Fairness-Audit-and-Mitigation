# src/models.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)


def build_preprocessor(X):
    """Builds preprocessing pipeline for numeric + categorical features."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def train_eval_logreg(X, y, test_size=0.2, random_state=42):
    """Train & evaluate Logistic Regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    preprocessor = build_preprocessor(X)
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=500))])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return evaluate_model("Logistic Regression", y_test, y_pred, y_prob)


def train_eval_rf(X, y, test_size=0.2, random_state=42):
    """Train & evaluate Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    preprocessor = build_preprocessor(X)
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=random_state))])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return evaluate_model("Random Forest", y_test, y_pred, y_prob)


def evaluate_model(name, y_true, y_pred, y_prob):
    """Print evaluation metrics for a model."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print("Confusion Matrix:\n", cm)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc, "confusion_matrix": cm}

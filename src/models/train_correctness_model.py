from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_features(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for predicting whether a student answers correctly.

    This is a lightweight model suitable for CPU-only training.
    """
    df = interactions.copy()
    df = df.sort_values(["student_id", "timestamp"]).reset_index(drop=True)

    df["prior_attempts"] = df.groupby("student_id").cumcount()

    df["student_prior_accuracy"] = (
        df.groupby("student_id")["correct"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(0.5)
    )

    df["skill_prior_accuracy"] = (
        df.groupby("skill_id")["correct"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(0.5)
    )

    df["question_prior_accuracy"] = (
        df.groupby("question_id")["correct"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(0.5)
    )

    df["hint_used_prev"] = (
        df.groupby("student_id")["hint_used"].shift(1).fillna(0).astype(float)
    )

    return df


def train_model(interactions: pd.DataFrame, output_path: str | Path | None = None):
    df = build_features(interactions)

    feature_cols = [
        "skill_id",
        "prior_attempts",
        "student_prior_accuracy",
        "skill_prior_accuracy",
        "question_prior_accuracy",
        "hint_used_prev",
    ]
    target_col = "correct"

    X = df[feature_cols]
    y = df[target_col].astype(int)

    numeric_features = [
        "prior_attempts",
        "student_prior_accuracy",
        "skill_prior_accuracy",
        "question_prior_accuracy",
        "hint_used_prev",
    ]
    categorical_features = ["skill_id"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }

    if y_test.nunique() == 2:
        proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)

    return model, metrics

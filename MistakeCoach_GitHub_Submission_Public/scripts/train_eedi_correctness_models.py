from __future__ import annotations

import argparse
import json
import time
from ast import literal_eval
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        classification_report,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. In this workspace, run with: "
        "PYTHONPATH=project/mistakecoach_ai_tutor/.deps "
        "/Users/yanmeiyi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 "
        "project/mistakecoach_ai_tutor/scripts/train_eedi_correctness_models.py"
    ) from exc


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
EEDI_DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = APP_ROOT / "data" / "processed" / "eedi_models"
MODEL_DIR = APP_ROOT / "models"
REPORT_PATH = APP_ROOT / "reports" / "eedi_model_comparison_report_zh.md"


TRAIN_DTYPES = {
    "QuestionId": "int32",
    "UserId": "int32",
    "AnswerId": "int64",
    "IsCorrect": "int8",
    "CorrectAnswer": "int8",
    "AnswerValue": "int8",
}


NUMERIC_FEATURES = [
    "student_attempts_prior",
    "student_accuracy_prior",
    "question_attempts_prior",
    "question_accuracy_prior",
    "subject_attempts_prior",
    "subject_accuracy_prior",
    "subject_count",
    "birth_year",
    "birth_year_missing",
    "premium_missing",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "PremiumPupil",
    "CorrectAnswer",
    "leaf_subject_id",
    "level1_subject_id",
]


def parse_subject_ids(value: object) -> list[int]:
    if pd.isna(value):
        return []
    try:
        parsed = literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [int(item) for item in parsed if pd.notna(item)]


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.1f}%"


def fmt(value: object, as_percent: bool = False) -> str:
    if as_percent:
        return pct(float(value)) if not pd.isna(value) else ""
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return ""
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if pd.isna(value):
        return ""
    return str(value)


def markdown_table(
    df: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    n: int = 10,
    percent_columns: set[str] | None = None,
) -> str:
    percent_columns = percent_columns or set()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.head(n).iterrows():
        lines.append("| " + " | ".join(fmt(row[col], col in percent_columns) for col in columns) + " |")
    return "\n".join(lines)


def read_summary_row_count(task: str) -> int | None:
    summary_path = APP_ROOT / "data" / "processed" / "eedi_eda" / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return int(summary.get("summary", {}).get("n_rows", 0)) or None


def sample_train_data(
    train_path: Path,
    sample_size: int,
    chunksize: int,
    random_state: int,
    total_rows: int | None,
) -> pd.DataFrame:
    if total_rows is None:
        total_rows = sum(1 for _ in open(train_path, "rb")) - 1

    frac = min(sample_size / total_rows * 1.08, 1.0)
    parts = []

    for chunk_no, chunk in enumerate(pd.read_csv(train_path, dtype=TRAIN_DTYPES, chunksize=chunksize), start=1):
        sampled = chunk.sample(frac=frac, random_state=random_state + chunk_no)
        parts.append(sampled)
        print(f"Sampled train chunk {chunk_no}: collected about {sum(len(p) for p in parts):,} rows", flush=True)

    sample = pd.concat(parts, ignore_index=True)
    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    else:
        sample = sample.reset_index(drop=True)
    return sample


def load_question_features(question_meta_path: Path) -> pd.DataFrame:
    question_meta = pd.read_csv(question_meta_path, encoding="utf-8-sig")
    subjects = question_meta["SubjectId"].map(parse_subject_ids)
    out = pd.DataFrame({"QuestionId": question_meta["QuestionId"]})
    out["subject_count"] = subjects.map(len).astype("int16")
    out["level1_subject_id"] = subjects.map(lambda values: values[1] if len(values) > 1 else -1)
    out["leaf_subject_id"] = subjects.map(lambda values: values[-1] if len(values) else -1)
    return out


def load_student_features(student_meta_path: Path) -> pd.DataFrame:
    student_meta = pd.read_csv(student_meta_path, encoding="utf-8-sig")
    out = student_meta[["UserId", "Gender", "PremiumPupil", "DateOfBirth"]].copy()
    out["birth_year"] = pd.to_datetime(out["DateOfBirth"], errors="coerce").dt.year
    out["birth_year_missing"] = out["birth_year"].isna().astype("int8")
    out["premium_missing"] = out["PremiumPupil"].isna().astype("int8")
    out = out.drop(columns=["DateOfBirth"])
    return out


def prepare_base_dataset(sample: pd.DataFrame, question_features: pd.DataFrame, student_features: pd.DataFrame) -> pd.DataFrame:
    df = sample.merge(question_features, on="QuestionId", how="left")
    df = df.merge(student_features, on="UserId", how="left")

    df["subject_count"] = df["subject_count"].fillna(0)
    df["level1_subject_id"] = df["level1_subject_id"].fillna(-1)
    df["leaf_subject_id"] = df["leaf_subject_id"].fillna(-1)
    df["birth_year_missing"] = df["birth_year_missing"].fillna(1)
    df["premium_missing"] = df["premium_missing"].fillna(1)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").fillna("missing")

    return df


def add_prior_rate_features(train_df: pd.DataFrame, test_df: pd.DataFrame, key: str, prefix: str, global_accuracy: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = train_df.groupby(key)["IsCorrect"].size()
    correct = train_df.groupby(key)["IsCorrect"].sum()

    train_count = train_df[key].map(counts).astype(float)
    train_correct = train_df[key].map(correct).astype(float)
    train_prior_count = train_count - 1
    train_df[f"{prefix}_attempts_prior"] = train_prior_count.clip(lower=0)
    train_df[f"{prefix}_accuracy_prior"] = np.where(
        train_prior_count > 0,
        (train_correct - train_df["IsCorrect"].astype(float)) / train_prior_count,
        global_accuracy,
    )

    test_count = test_df[key].map(counts).fillna(0).astype(float)
    test_correct = test_df[key].map(correct).fillna(0).astype(float)
    test_df[f"{prefix}_attempts_prior"] = test_count
    test_df[f"{prefix}_accuracy_prior"] = np.where(
        test_count > 0,
        test_correct / test_count,
        global_accuracy,
    )

    return train_df, test_df


def add_all_prior_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    global_accuracy = float(train_df["IsCorrect"].mean())
    train_df, test_df = add_prior_rate_features(train_df, test_df, "UserId", "student", global_accuracy)
    train_df, test_df = add_prior_rate_features(train_df, test_df, "QuestionId", "question", global_accuracy)
    train_df, test_df = add_prior_rate_features(train_df, test_df, "leaf_subject_id", "subject", global_accuracy)
    return train_df, test_df, global_accuracy


def build_preprocessors() -> tuple[ColumnTransformer, ColumnTransformer, ColumnTransformer, ColumnTransformer]:
    onehot = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    ordinal = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    numeric_only = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ]
    )
    numeric_scaled = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ]
    )
    return onehot, ordinal, numeric_only, numeric_scaled


def build_models(random_state: int) -> dict[str, Pipeline]:
    onehot, ordinal, numeric_only, numeric_scaled = build_preprocessors()
    return {
        "baseline_most_frequent": Pipeline(
            steps=[("clf", DummyClassifier(strategy="most_frequent"))]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", onehot),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        solver="saga",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            steps=[
                ("preprocess", onehot),
                (
                    "clf",
                    SGDClassifier(
                        loss="hinge",
                        alpha=0.0001,
                        max_iter=1000,
                        tol=1e-3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("preprocess", ordinal),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=8,
                        min_samples_leaf=200,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", ordinal),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=80,
                        max_depth=14,
                        min_samples_leaf=50,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocess", numeric_only),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=120,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.8,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "neural_network_mlp": Pipeline(
            steps=[
                ("preprocess", numeric_scaled),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        alpha=0.001,
                        learning_rate_init=0.001,
                        max_iter=50,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=5,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gaussian_naive_bayes": Pipeline(
            steps=[
                ("preprocess", ordinal),
                ("clf", GaussianNB()),
            ]
        ),
    }


def prediction_outputs(model: Pipeline, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    preds = model.predict(x)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[:, 1]
        return preds, proba, proba
    if hasattr(model, "decision_function"):
        score = model.decision_function(x)
        return preds, score, None
    return preds, preds.astype(float), None


def evaluate_model(name: str, model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, object], Pipeline]:
    started = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - started

    preds, score, proba = prediction_outputs(model, x_test)

    row = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, score)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1])) if proba is not None else np.nan,
        "brier_score": float(brier_score_loss(y_test, proba)) if proba is not None else np.nan,
        "train_seconds": float(train_seconds),
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    }
    print(
        f"{name}: accuracy={row['accuracy']:.4f}, auc={row['roc_auc']:.4f}, "
        f"f1={row['f1']:.4f}, seconds={row['train_seconds']:.1f}",
        flush=True,
    )
    return row, model


def extract_feature_importance(best_model: Pipeline, model_name: str) -> pd.DataFrame:
    clf = best_model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return pd.DataFrame()

    importances = clf.feature_importances_
    preprocessor = best_model.named_steps.get("preprocess")
    if preprocessor is not None:
        try:
            feature_names = [
                name.split("__", 1)[-1]
                for name in preprocessor.get_feature_names_out()
            ]
        except Exception:
            feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    else:
        feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    importance["model"] = model_name
    return importance


def write_report(
    metrics_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    output_dir: Path,
    model_path: Path,
    sample_size: int,
    train_size: int,
    test_size: int,
    global_accuracy: float,
) -> None:
    best = metrics_df.sort_values(["roc_auc", "accuracy"], ascending=[False, False]).iloc[0]
    report = f"""# Eedi Correctness Prediction Model Comparison Report

## 1. Modeling Goal

This modeling task predicts `IsCorrect`, meaning whether a student will answer a given math diagnostic question correctly. The feature set excludes the student's actual selected `AnswerValue` to avoid target leakage.

## 2. Data and Split

- Sampled records: {fmt(sample_size)}
- Training set: {fmt(train_size)}
- Test set: {fmt(test_size)}
- Training global accuracy: {pct(global_accuracy)}
- Split method: stratified random split with a 25% test set

Historical accuracy features are calculated only from the training set. Within the training set, student, question, and subject historical accuracies use leave-one-out estimates to reduce leakage. Logistic regression and linear SVM use one-hot encoding for categorical variables; tree, random forest, and naive Bayes use ordinal encoding; gradient boosting and neural network MLP use numeric historical features to avoid imposing artificial ordering on categorical variables. Linear SVM outputs a decision score, so ROC-AUC is available, but probability-based log loss and Brier score are not.

## 3. Model Comparison

{markdown_table(metrics_df, ["model", "accuracy", "roc_auc", "f1", "precision", "recall", "log_loss", "brier_score", "train_seconds"], ["Model", "Accuracy", "ROC-AUC", "F1", "Precision", "Recall", "Log Loss", "Brier", "Train Seconds"], len(metrics_df), {"accuracy", "roc_auc", "f1", "precision", "recall"})}

The best model is selected by ROC-AUC: **{best["model"]}**, with ROC-AUC {pct(best["roc_auc"])} and accuracy {pct(best["accuracy"])}.

## 4. Feature Importance

"""
    if len(feature_importance):
        report += markdown_table(
            feature_importance,
            ["feature", "importance"],
            ["Feature", "Importance"],
            15,
        )
    else:
        report += "The best model does not provide direct feature importance output."

    report += f"""

## 5. Initial Conclusions

1. The baseline predicts only the majority class, so its accuracy is close to the overall correctness rate, but F1 and ROC-AUC provide limited information.
2. Student historical accuracy, question historical accuracy, and subject-level historical accuracy are the core predictive signals.
3. Tree-based methods can capture nonlinear relationships and are strong candidates for the recommendation component.
4. Logistic regression remains useful as an interpretable model because it is fast and easier to explain.
5. Linear SVM serves as a margin-based classifier comparison; even when it is not the best model, it helps contrast linear separation with probability-based models.
6. Neural network MLP tests whether nonlinear combinations of numeric features improve prediction quality.
7. The next step is to connect the best model's predicted probabilities to mastery scoring and next-question recommendation logic.

## 6. Output Files

The modeling outputs are saved in `{output_dir}`:

- `model_metrics.csv`
- `model_metrics.json`
- `modeling_features_sample.csv`
- `feature_importance.csv`

The best model is saved to `{model_path}`.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Eedi correctness prediction models.")
    parser.add_argument("--task", choices=["1_2", "3_4"], default="1_2")
    parser.add_argument("--sample-size", type=int, default=300_000)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_path = EEDI_DATA_DIR / "train_data" / f"train_task_{args.task}.csv"
    question_meta_path = EEDI_DATA_DIR / "metadata" / f"question_metadata_task_{args.task}.csv"
    student_meta_path = EEDI_DATA_DIR / "metadata" / f"student_metadata_task_{args.task}.csv"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = read_summary_row_count(args.task)
    sample = sample_train_data(train_path, args.sample_size, args.chunksize, args.random_state, total_rows)
    question_features = load_question_features(question_meta_path)
    student_features = load_student_features(student_meta_path)
    base = prepare_base_dataset(sample, question_features, student_features)

    train_idx, test_idx = train_test_split(
        np.arange(len(base)),
        test_size=0.25,
        random_state=args.random_state,
        stratify=base["IsCorrect"],
    )
    train_df = base.iloc[train_idx].copy()
    test_df = base.iloc[test_idx].copy()
    train_df, test_df, global_accuracy = add_all_prior_features(train_df, test_df)

    feature_df = pd.concat(
        [
            train_df.assign(split="train"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    feature_df[
        ["split", "UserId", "QuestionId", "IsCorrect", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    ].to_csv(OUTPUT_DIR / "modeling_features_sample.csv", index=False)

    x_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df["IsCorrect"].astype(int)
    x_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df["IsCorrect"].astype(int)

    metrics = []
    trained_models = {}
    for name, model in build_models(args.random_state).items():
        row, fitted = evaluate_model(name, model, x_train, y_train, x_test, y_test)
        metrics.append(row)
        trained_models[name] = fitted

    metrics_df = pd.DataFrame(metrics).sort_values(["roc_auc", "accuracy"], ascending=[False, False])
    reportable_metrics = metrics_df.drop(columns=["classification_report"])
    reportable_metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    metrics_json = metrics_df.to_dict(orient="records")
    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    best_model_path = MODEL_DIR / f"eedi_correctness_{best_model_name}.joblib"
    joblib.dump(best_model, best_model_path)

    feature_importance = extract_feature_importance(best_model, best_model_name)
    if len(feature_importance):
        feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    else:
        pd.DataFrame(columns=["feature", "importance", "model"]).to_csv(
            OUTPUT_DIR / "feature_importance.csv",
            index=False,
        )

    write_report(
        reportable_metrics,
        feature_importance,
        OUTPUT_DIR,
        best_model_path,
        len(base),
        len(train_df),
        len(test_df),
        global_accuracy,
    )

    print(f"Model outputs saved to: {OUTPUT_DIR}", flush=True)
    print(f"Model comparison report saved to: {REPORT_PATH}", flush=True)
    print(f"Best model saved to: {best_model_path}", flush=True)


if __name__ == "__main__":
    main()

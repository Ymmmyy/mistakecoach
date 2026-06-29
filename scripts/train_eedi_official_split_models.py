from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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

from train_eedi_correctness_models import (
    APP_ROOT,
    CATEGORICAL_FEATURES,
    EEDI_DATA_DIR,
    MODEL_DIR,
    NUMERIC_FEATURES,
    TRAIN_DTYPES,
    build_models,
    extract_feature_importance,
    fmt,
    load_question_features,
    load_student_features,
    markdown_table,
    pct,
    prediction_outputs,
    read_summary_row_count,
)


OUTPUT_DIR = APP_ROOT / "data" / "processed" / "eedi_official_split_models"
REPORT_PATH = APP_ROOT / "reports" / "eedi_official_split_model_report_zh.md"


def add_series(left: pd.Series | None, right: pd.Series) -> pd.Series:
    if left is None:
        return right.copy()
    return left.add(right, fill_value=0)


def scan_train_for_sample_and_stats(
    train_path: Path,
    question_features: pd.DataFrame,
    train_sample_size: int,
    chunksize: int,
    random_state: int,
    total_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if total_rows is None:
        total_rows = sum(1 for _ in open(train_path, "rb")) - 1

    sample_frac = min(train_sample_size / total_rows * 1.08, 1.0)
    question_leaf = question_features.set_index("QuestionId")["leaf_subject_id"]

    n_rows = 0
    correct_sum = 0
    samples = []

    user_counts = None
    user_correct = None
    question_counts = None
    question_correct = None
    subject_counts = None
    subject_correct = None
    correct_answer_counts = None

    for chunk_no, chunk in enumerate(pd.read_csv(train_path, dtype=TRAIN_DTYPES, chunksize=chunksize), start=1):
        n_rows += len(chunk)
        correct_sum += int(chunk["IsCorrect"].sum())

        leaf_subject = chunk["QuestionId"].map(question_leaf).fillna(-1).astype("int64")
        chunk_for_subject = chunk.assign(leaf_subject_id=leaf_subject)

        user_counts = add_series(user_counts, chunk.groupby("UserId").size())
        user_correct = add_series(user_correct, chunk.groupby("UserId")["IsCorrect"].sum())
        question_counts = add_series(question_counts, chunk.groupby("QuestionId").size())
        question_correct = add_series(question_correct, chunk.groupby("QuestionId")["IsCorrect"].sum())
        subject_counts = add_series(subject_counts, chunk_for_subject.groupby("leaf_subject_id").size())
        subject_correct = add_series(subject_correct, chunk_for_subject.groupby("leaf_subject_id")["IsCorrect"].sum())
        correct_answer_counts = add_series(
            correct_answer_counts,
            chunk.groupby(["QuestionId", "CorrectAnswer"]).size(),
        )

        sampled = chunk.sample(frac=sample_frac, random_state=random_state + chunk_no)
        samples.append(sampled)
        print(
            f"Scanned train chunk {chunk_no}: {n_rows:,} rows; sampled about {sum(len(part) for part in samples):,}",
            flush=True,
        )

    train_sample = pd.concat(samples, ignore_index=True)
    if len(train_sample) > train_sample_size:
        train_sample = train_sample.sample(n=train_sample_size, random_state=random_state).reset_index(drop=True)
    else:
        train_sample = train_sample.reset_index(drop=True)

    correct_answer_map = (
        correct_answer_counts.astype("int64")
        .rename("count")
        .reset_index()
        .sort_values(["QuestionId", "count"], ascending=[True, False])
        .drop_duplicates("QuestionId")
        .set_index("QuestionId")["CorrectAnswer"]
    )

    stats = {
        "n_rows": int(n_rows),
        "global_accuracy": float(correct_sum / n_rows),
        "user_counts": user_counts.astype("int64"),
        "user_correct": user_correct.astype("int64"),
        "question_counts": question_counts.astype("int64"),
        "question_correct": question_correct.astype("int64"),
        "subject_counts": subject_counts.astype("int64"),
        "subject_correct": subject_correct.astype("int64"),
        "correct_answer_map": correct_answer_map.astype("int8"),
    }
    return train_sample, stats


def make_feature_frame(
    df: pd.DataFrame,
    question_features: pd.DataFrame,
    student_features: pd.DataFrame,
    correct_answer_map: pd.Series,
) -> pd.DataFrame:
    keep_cols = ["QuestionId", "UserId", "AnswerId", "IsCorrect"]
    out = df[keep_cols].copy()
    if "CorrectAnswer" in df.columns:
        out["CorrectAnswer"] = df["CorrectAnswer"]
    else:
        out["CorrectAnswer"] = out["QuestionId"].map(correct_answer_map)

    out = out.merge(question_features, on="QuestionId", how="left")
    out = out.merge(student_features, on="UserId", how="left")

    out["CorrectAnswer"] = out["CorrectAnswer"].fillna(-1)
    out["subject_count"] = out["subject_count"].fillna(0)
    out["level1_subject_id"] = out["level1_subject_id"].fillna(-1)
    out["leaf_subject_id"] = out["leaf_subject_id"].fillna(-1)
    out["birth_year_missing"] = out["birth_year_missing"].fillna(1)
    out["premium_missing"] = out["premium_missing"].fillna(1)
    return out


def add_prior_from_full_train(
    df: pd.DataFrame,
    key: str,
    counts: pd.Series,
    correct: pd.Series,
    prefix: str,
    global_accuracy: float,
    subtract_current: bool,
) -> pd.DataFrame:
    mapped_count = df[key].map(counts).fillna(0).astype(float)
    mapped_correct = df[key].map(correct).fillna(0).astype(float)

    if subtract_current:
        prior_count = (mapped_count - 1).clip(lower=0)
        prior_correct = (mapped_correct - df["IsCorrect"].astype(float)).clip(lower=0)
    else:
        prior_count = mapped_count
        prior_correct = mapped_correct

    df[f"{prefix}_attempts_prior"] = prior_count
    df[f"{prefix}_accuracy_prior"] = np.where(
        prior_count > 0,
        prior_correct / prior_count,
        global_accuracy,
    )
    return df


def add_all_prior_features_from_full_train(
    df: pd.DataFrame,
    stats: dict[str, object],
    subtract_current: bool,
) -> pd.DataFrame:
    global_accuracy = float(stats["global_accuracy"])
    df = add_prior_from_full_train(
        df,
        "UserId",
        stats["user_counts"],
        stats["user_correct"],
        "student",
        global_accuracy,
        subtract_current,
    )
    df = add_prior_from_full_train(
        df,
        "QuestionId",
        stats["question_counts"],
        stats["question_correct"],
        "question",
        global_accuracy,
        subtract_current,
    )
    df = add_prior_from_full_train(
        df,
        "leaf_subject_id",
        stats["subject_counts"],
        stats["subject_correct"],
        "subject",
        global_accuracy,
        subtract_current,
    )

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").fillna("missing")
    return df


def load_test_split(path: Path, limit: int | None) -> pd.DataFrame:
    dtypes = {
        "QuestionId": "int32",
        "UserId": "int32",
        "AnswerId": "int64",
        "IsCorrect": "int8",
    }
    return pd.read_csv(path, dtype=dtypes, nrows=limit)


def coverage_for_split(df: pd.DataFrame, stats: dict[str, object], split_name: str) -> dict[str, object]:
    known_users = df["UserId"].isin(stats["user_counts"].index)
    known_questions = df["QuestionId"].isin(stats["question_counts"].index)
    leaf_subject_id = pd.to_numeric(df["leaf_subject_id"], errors="coerce")
    known_subjects = leaf_subject_id.isin(stats["subject_counts"].index)
    known_answers = df["CorrectAnswer"].astype(float).ge(0)
    return {
        "split": split_name,
        "rows": int(len(df)),
        "accuracy": float(df["IsCorrect"].mean()),
        "known_user_share": float(known_users.mean()),
        "known_question_share": float(known_questions.mean()),
        "known_subject_share": float(known_subjects.mean()),
        "known_correct_answer_share": float(known_answers.mean()),
    }


def evaluate_fitted_model(
    name: str,
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    split_name: str,
    train_seconds: float,
) -> dict[str, object]:
    started = time.perf_counter()
    preds, score, proba = prediction_outputs(model, x_test)
    predict_seconds = time.perf_counter() - started

    return {
        "model": name,
        "split": split_name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, score)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1])) if proba is not None else np.nan,
        "brier_score": float(brier_score_loss(y_test, proba)) if proba is not None else np.nan,
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    }


def write_predictions(model, df: pd.DataFrame, split_name: str, output_dir: Path) -> None:
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preds, score, proba = prediction_outputs(model, x)
    predictions = df[["QuestionId", "UserId", "AnswerId", "IsCorrect"]].copy()
    if proba is not None:
        predictions["predicted_probability_correct"] = proba
    else:
        predictions["decision_score_correct"] = score
    predictions["predicted_IsCorrect"] = preds
    predictions.to_csv(output_dir / f"{split_name}_best_model_predictions.csv", index=False)


def write_report(
    metrics_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    best_model_name: str,
    train_sample_size: int,
    full_train_rows: int,
    global_accuracy: float,
    model_path: Path,
) -> None:
    reportable = metrics_df.drop(columns=["classification_report"], errors="ignore")
    test_metrics = reportable.loc[reportable["split"] == "test"].sort_values(
        ["roc_auc", "accuracy"], ascending=[False, False]
    )

    report = f"""# Eedi Train/Test Correctness Prediction Report

## 1. Evaluation Setup

This evaluation uses the dataset-provided train/test split rather than a random split. The model is trained from `train_data/train_task_1_2.csv` and evaluated on `test_data/test_public_answers_task_1.csv`.

- full train rows：{fmt(full_train_rows)}
- train sample used for model fitting：{fmt(train_sample_size)}
- full train global accuracy：{pct(global_accuracy)}
- test rows：{fmt(int(coverage_df.loc[coverage_df["split"] == "test", "rows"].iloc[0]))}

Historical features such as student/question/subject prior accuracy are computed from the full training data only. The test set is not used for feature statistics or model training.
Linear SVM uses decision scores for ROC-AUC and does not report probability metrics such as log loss or Brier score.

## 2. Test Set Coverage

{markdown_table(coverage_df, ["split", "rows", "accuracy", "known_user_share", "known_question_share", "known_subject_share", "known_correct_answer_share"], ["Split", "Rows", "True Accuracy", "Known Users", "Known Questions", "Known Subjects", "Known Correct Answers"], len(coverage_df), {"accuracy", "known_user_share", "known_question_share", "known_subject_share", "known_correct_answer_share"})}

## 3. Model Results

{markdown_table(test_metrics, ["model", "accuracy", "roc_auc", "f1", "precision", "recall", "log_loss", "brier_score", "train_seconds", "predict_seconds"], ["Model", "Accuracy", "ROC-AUC", "F1", "Precision", "Recall", "Log Loss", "Brier", "Train Seconds", "Predict Seconds"], len(test_metrics), {"accuracy", "roc_auc", "f1", "precision", "recall"})}

The best model by test ROC-AUC is **{best_model_name}**.

## 4. Feature Importance

"""
    if len(feature_importance):
        report += markdown_table(feature_importance, ["feature", "importance"], ["Feature", "Importance"], 15)
    else:
        report += "The best model does not provide direct feature importance."

    report += f"""

## 5. Conclusion

1. This evaluation is suitable for the final report because train and test come from the dataset-provided split.
2. The test-set accuracy can be compared with the training global accuracy to assess distribution consistency.
3. Student and question prior accuracy are strong signals when test coverage is high.
4. The best model is saved to `{model_path}`.

## 6. Output Files

Output directory: `{OUTPUT_DIR}`

- `public_test_model_metrics.csv`
- `public_test_coverage.csv`
- `official_split_feature_importance.csv`
- `public_best_model_predictions.csv`
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on Eedi train data and evaluate on official test data.")
    parser.add_argument("--train-sample-size", type=int, default=300_000)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_path = EEDI_DATA_DIR / "train_data" / "train_task_1_2.csv"
    question_meta_path = EEDI_DATA_DIR / "metadata" / "question_metadata_task_1_2.csv"
    student_meta_path = EEDI_DATA_DIR / "metadata" / "student_metadata_task_1_2.csv"
    test_path = EEDI_DATA_DIR / "test_data" / "test_public_answers_task_1.csv"

    question_features = load_question_features(question_meta_path)
    student_features = load_student_features(student_meta_path)
    total_rows = read_summary_row_count("1_2")

    train_sample, stats = scan_train_for_sample_and_stats(
        train_path,
        question_features,
        args.train_sample_size,
        args.chunksize,
        args.random_state,
        total_rows,
    )

    train_df = make_feature_frame(
        train_sample,
        question_features,
        student_features,
        stats["correct_answer_map"],
    )
    train_df = add_all_prior_features_from_full_train(train_df, stats, subtract_current=True)

    test_df = make_feature_frame(
        load_test_split(test_path, args.test_limit),
        question_features,
        student_features,
        stats["correct_answer_map"],
    )
    test_df = add_all_prior_features_from_full_train(test_df, stats, subtract_current=False)

    coverage_df = pd.DataFrame(
        [
            coverage_for_split(test_df, stats, "test"),
        ]
    )
    coverage_df.to_csv(OUTPUT_DIR / "public_test_coverage.csv", index=False)

    x_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df["IsCorrect"].astype(int)
    tests = {
        "test": (
            test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
            test_df["IsCorrect"].astype(int),
        ),
    }

    metrics = []
    trained_models = {}
    for name, model in build_models(args.random_state).items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - started
        trained_models[name] = model

        for split_name, (x_test, y_test) in tests.items():
            row = evaluate_fitted_model(name, model, x_test, y_test, split_name, train_seconds)
            metrics.append(row)
            print(
                f"{name} on {split_name}: accuracy={row['accuracy']:.4f}, "
                f"auc={row['roc_auc']:.4f}, f1={row['f1']:.4f}",
                flush=True,
            )

    metrics_df = pd.DataFrame(metrics)
    reportable_metrics = metrics_df.drop(columns=["classification_report"])
    reportable_metrics.to_csv(OUTPUT_DIR / "public_test_model_metrics.csv", index=False)
    with open(OUTPUT_DIR / "public_test_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    test_ranking = reportable_metrics.loc[reportable_metrics["split"] == "test"].sort_values(
        ["roc_auc", "accuracy"], ascending=[False, False]
    )
    best_model_name = str(test_ranking.iloc[0]["model"])
    best_model = trained_models[best_model_name]
    best_model_path = MODEL_DIR / f"eedi_official_split_correctness_{best_model_name}.joblib"
    joblib.dump(best_model, best_model_path)

    feature_importance = extract_feature_importance(best_model, best_model_name)
    if len(feature_importance):
        feature_importance.to_csv(OUTPUT_DIR / "official_split_feature_importance.csv", index=False)
    else:
        pd.DataFrame(columns=["feature", "importance", "model"]).to_csv(
            OUTPUT_DIR / "official_split_feature_importance.csv",
            index=False,
        )

    write_predictions(best_model, test_df, "public", OUTPUT_DIR)

    write_report(
        reportable_metrics,
        coverage_df,
        feature_importance,
        best_model_name,
        len(train_df),
        int(stats["n_rows"]),
        float(stats["global_accuracy"]),
        best_model_path,
    )

    print(f"Official split outputs saved to: {OUTPUT_DIR}", flush=True)
    print(f"Official split report saved to: {REPORT_PATH}", flush=True)
    print(f"Best model saved to: {best_model_path}", flush=True)


if __name__ == "__main__":
    main()

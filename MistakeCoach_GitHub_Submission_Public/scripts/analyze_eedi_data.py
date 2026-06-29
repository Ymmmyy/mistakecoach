from __future__ import annotations

import argparse
import json
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
EEDI_DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = APP_ROOT / "data" / "processed" / "eedi_eda"
REPORT_PATH = APP_ROOT / "reports" / "eedi_eda_report_zh.md"


TRAIN_DTYPES = {
    "QuestionId": "int32",
    "UserId": "int32",
    "AnswerId": "int64",
    "IsCorrect": "int8",
    "CorrectAnswer": "int8",
    "AnswerValue": "int8",
}


def add_series(left: pd.Series | None, right: pd.Series) -> pd.Series:
    if left is None:
        return right.copy()
    return left.add(right, fill_value=0)


def parse_subject_ids(value: object) -> list[int]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
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
        return f"{value:,.2f}"
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
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.head(n).iterrows():
        out.append("| " + " | ".join(fmt(row[col], col in percent_columns) for col in columns) + " |")
    return "\n".join(out)


def make_bins(series: pd.Series, bins: list[float], labels: list[str]) -> pd.DataFrame:
    binned = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    return (
        binned.value_counts(sort=False)
        .rename_axis("bin")
        .reset_index(name="count")
    )


def analyze_train(train_path: Path, chunksize: int, max_chunks: int | None) -> dict[str, object]:
    n_rows = 0
    correct_sum = 0
    user_attempts = None
    user_correct = None
    question_attempts = None
    question_correct = None
    answer_value_counts = None
    correct_answer_counts = None
    confusion_counts = None
    wrong_answer_counts = None
    question_wrong_answer_counts = None

    reader = pd.read_csv(train_path, dtype=TRAIN_DTYPES, chunksize=chunksize)

    for chunk_no, chunk in enumerate(reader, start=1):
        if max_chunks is not None and chunk_no > max_chunks:
            break

        n_rows += len(chunk)
        correct_sum += int(chunk["IsCorrect"].sum())

        user_attempts = add_series(user_attempts, chunk.groupby("UserId").size())
        user_correct = add_series(user_correct, chunk.groupby("UserId")["IsCorrect"].sum())

        question_attempts = add_series(question_attempts, chunk.groupby("QuestionId").size())
        question_correct = add_series(question_correct, chunk.groupby("QuestionId")["IsCorrect"].sum())

        answer_value_counts = add_series(answer_value_counts, chunk["AnswerValue"].value_counts())
        correct_answer_counts = add_series(correct_answer_counts, chunk["CorrectAnswer"].value_counts())
        confusion_counts = add_series(
            confusion_counts,
            chunk.groupby(["CorrectAnswer", "AnswerValue"]).size(),
        )

        wrong = chunk.loc[chunk["IsCorrect"] == 0, ["QuestionId", "AnswerValue"]]
        wrong_answer_counts = add_series(wrong_answer_counts, wrong["AnswerValue"].value_counts())
        question_wrong_answer_counts = add_series(
            question_wrong_answer_counts,
            wrong.groupby(["QuestionId", "AnswerValue"]).size(),
        )

        print(f"Processed train chunk {chunk_no}: {n_rows:,} rows", flush=True)

    student_stats = pd.DataFrame(
        {
            "attempts": user_attempts.astype("int64"),
            "correct": user_correct.astype("int64"),
        }
    ).rename_axis("UserId").reset_index()
    student_stats["accuracy"] = student_stats["correct"] / student_stats["attempts"]
    student_stats = student_stats.sort_values(["attempts", "accuracy"], ascending=[False, False])

    question_stats = pd.DataFrame(
        {
            "attempts": question_attempts.astype("int64"),
            "correct": question_correct.astype("int64"),
        }
    ).rename_axis("QuestionId").reset_index()
    question_stats["wrong"] = question_stats["attempts"] - question_stats["correct"]
    question_stats["accuracy"] = question_stats["correct"] / question_stats["attempts"]
    question_stats["difficulty"] = 1 - question_stats["accuracy"]
    question_stats = question_stats.sort_values(["difficulty", "attempts"], ascending=[False, False])

    answer_summary = pd.DataFrame(
        {
            "selected_count": answer_value_counts.astype("int64"),
            "correct_answer_count": correct_answer_counts.astype("int64"),
            "wrong_selected_count": wrong_answer_counts.astype("int64"),
        }
    ).fillna(0).astype("int64").rename_axis("answer_value").reset_index()
    answer_summary["selected_share"] = answer_summary["selected_count"] / n_rows
    answer_summary["wrong_selected_share"] = answer_summary["wrong_selected_count"] / max(n_rows - correct_sum, 1)
    answer_summary = answer_summary.sort_values("answer_value")

    confusion = (
        confusion_counts.astype("int64")
        .rename("count")
        .reset_index()
        .sort_values(["CorrectAnswer", "AnswerValue"])
    )

    question_wrong_patterns = (
        question_wrong_answer_counts.astype("int64")
        .rename("wrong_count")
        .reset_index()
        .sort_values(["QuestionId", "wrong_count"], ascending=[True, False])
    )
    question_wrong_patterns = question_wrong_patterns.drop_duplicates("QuestionId")
    question_wrong_patterns = question_wrong_patterns.merge(
        question_stats[["QuestionId", "wrong", "attempts", "accuracy"]],
        on="QuestionId",
        how="left",
    )
    question_wrong_patterns["top_wrong_share_of_wrong"] = (
        question_wrong_patterns["wrong_count"] / question_wrong_patterns["wrong"].replace(0, np.nan)
    )
    question_wrong_patterns = question_wrong_patterns.sort_values(
        ["wrong_count", "top_wrong_share_of_wrong"], ascending=[False, False]
    )

    student_attempt_bins = make_bins(
        student_stats["attempts"],
        [0, 10, 25, 50, 100, 200, 500, 1000, np.inf],
        ["1-10", "11-25", "26-50", "51-100", "101-200", "201-500", "501-1000", "1000+"],
    )
    question_accuracy_bins = make_bins(
        question_stats["accuracy"],
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"],
    )

    return {
        "summary": {
            "n_rows": int(n_rows),
            "n_students": int(student_stats["UserId"].nunique()),
            "n_questions": int(question_stats["QuestionId"].nunique()),
            "n_correct": int(correct_sum),
            "overall_accuracy": float(correct_sum / n_rows),
            "student_attempt_median": float(student_stats["attempts"].median()),
            "student_attempt_q1": float(student_stats["attempts"].quantile(0.25)),
            "student_attempt_q3": float(student_stats["attempts"].quantile(0.75)),
            "question_attempt_median": float(question_stats["attempts"].median()),
            "question_attempt_q1": float(question_stats["attempts"].quantile(0.25)),
            "question_attempt_q3": float(question_stats["attempts"].quantile(0.75)),
            "question_accuracy_median": float(question_stats["accuracy"].median()),
            "question_accuracy_q1": float(question_stats["accuracy"].quantile(0.25)),
            "question_accuracy_q3": float(question_stats["accuracy"].quantile(0.75)),
            "questions_below_30_accuracy_share": float((question_stats["accuracy"] < 0.30).mean()),
            "questions_above_70_accuracy_share": float((question_stats["accuracy"] > 0.70).mean()),
        },
        "student_stats": student_stats,
        "question_stats": question_stats,
        "answer_summary": answer_summary,
        "confusion": confusion,
        "question_wrong_patterns": question_wrong_patterns,
        "student_attempt_bins": student_attempt_bins,
        "question_accuracy_bins": question_accuracy_bins,
    }


def analyze_question_subjects(question_stats: pd.DataFrame, question_meta_path: Path, subject_meta_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    question_meta = pd.read_csv(question_meta_path, encoding="utf-8-sig")
    question_subjects = question_meta.copy()
    question_subjects["SubjectId"] = question_subjects["SubjectId"].map(parse_subject_ids)
    question_subjects["subject_count"] = question_subjects["SubjectId"].map(len)
    exploded = question_subjects[["QuestionId", "SubjectId"]].explode("SubjectId")
    exploded = exploded.dropna(subset=["SubjectId"]).copy()
    exploded["SubjectId"] = exploded["SubjectId"].astype("int64")

    subject_meta = pd.read_csv(subject_meta_path, encoding="utf-8-sig")
    subject_meta["ParentId"] = subject_meta["ParentId"].replace("NULL", np.nan)
    subject_meta["ParentId"] = pd.to_numeric(subject_meta["ParentId"], errors="coerce")

    question_with_subject = exploded.merge(question_stats, on="QuestionId", how="inner")
    subject_stats = (
        question_with_subject.groupby("SubjectId")
        .agg(
            n_questions=("QuestionId", "nunique"),
            attempts=("attempts", "sum"),
            correct=("correct", "sum"),
        )
        .reset_index()
    )
    subject_stats["accuracy"] = subject_stats["correct"] / subject_stats["attempts"]
    subject_stats["difficulty"] = 1 - subject_stats["accuracy"]
    subject_stats = subject_stats.merge(subject_meta, on="SubjectId", how="left")
    subject_stats = subject_stats.sort_values(["difficulty", "attempts"], ascending=[False, False])

    question_subject_counts = (
        question_subjects[["QuestionId", "subject_count"]]
        .merge(question_stats[["QuestionId", "attempts", "accuracy"]], on="QuestionId", how="right")
    )

    return subject_stats, question_subject_counts


def analyze_student_metadata(student_stats: pd.DataFrame, student_meta_path: Path) -> dict[str, pd.DataFrame]:
    student_meta = pd.read_csv(student_meta_path, encoding="utf-8-sig")
    missing = (
        student_meta.isna()
        .mean()
        .rename("missing_share")
        .reset_index()
        .rename(columns={"index": "field"})
    )
    missing["missing_count"] = student_meta.isna().sum().values

    enriched = student_stats.merge(student_meta, on="UserId", how="left")
    gender = (
        enriched.groupby("Gender", dropna=False)
        .agg(students=("UserId", "nunique"), attempts=("attempts", "sum"), correct=("correct", "sum"))
        .reset_index()
    )
    gender["accuracy"] = gender["correct"] / gender["attempts"]

    premium = (
        enriched.groupby("PremiumPupil", dropna=False)
        .agg(students=("UserId", "nunique"), attempts=("attempts", "sum"), correct=("correct", "sum"))
        .reset_index()
    )
    premium["accuracy"] = premium["correct"] / premium["attempts"]

    enriched["birth_year"] = pd.to_datetime(enriched["DateOfBirth"], errors="coerce").dt.year
    birth_year = (
        enriched.dropna(subset=["birth_year"])
        .groupby("birth_year")
        .agg(students=("UserId", "nunique"), attempts=("attempts", "sum"), correct=("correct", "sum"))
        .reset_index()
    )
    birth_year["birth_year"] = birth_year["birth_year"].astype("int64")
    birth_year["accuracy"] = birth_year["correct"] / birth_year["attempts"]
    birth_year = birth_year.sort_values("birth_year")

    return {
        "student_metadata_missing": missing,
        "accuracy_by_gender": gender,
        "accuracy_by_premium": premium,
        "accuracy_by_birth_year": birth_year,
    }


def analyze_answer_metadata(answer_meta_path: Path, chunksize: int, max_chunks: int | None) -> dict[str, object]:
    n_rows = 0
    missing_counts = None
    month_counts = None
    hour_counts = None
    date_min = None
    date_max = None

    for chunk_no, chunk in enumerate(pd.read_csv(answer_meta_path, chunksize=chunksize), start=1):
        if max_chunks is not None and chunk_no > max_chunks:
            break

        n_rows += len(chunk)
        missing_counts = add_series(missing_counts, chunk.isna().sum())

        date_text = chunk["DateAnswered"].dropna().astype(str)
        if len(date_text):
            current_min = date_text.min()
            current_max = date_text.max()
            date_min = current_min if date_min is None else min(date_min, current_min)
            date_max = current_max if date_max is None else max(date_max, current_max)
            month_counts = add_series(month_counts, date_text.str.slice(0, 7).value_counts())
            hour_counts = add_series(hour_counts, date_text.str.slice(11, 13).value_counts())

        print(f"Processed answer metadata chunk {chunk_no}: {n_rows:,} rows", flush=True)

    missing = (
        (missing_counts / n_rows)
        .rename("missing_share")
        .reset_index()
        .rename(columns={"index": "field"})
    )
    missing["missing_count"] = missing_counts.astype("int64").values

    month_summary = (
        month_counts.astype("int64")
        .rename_axis("month")
        .reset_index(name="answers")
        .sort_values("month")
    )
    hour_summary = (
        hour_counts.astype("int64")
        .rename_axis("hour")
        .reset_index(name="answers")
        .sort_values("hour")
    )

    return {
        "answer_metadata_summary": {
            "n_rows": int(n_rows),
            "date_min": date_min,
            "date_max": date_max,
        },
        "answer_metadata_missing": missing,
        "answers_by_month": month_summary,
        "answers_by_hour": hour_summary,
    }


def write_outputs(results: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_ready = {
        key: value
        for key, value in results.items()
        if isinstance(value, dict) and not any(isinstance(v, pd.DataFrame) for v in value.values())
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2, ensure_ascii=False)

    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"{key}.csv", index=False)
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, pd.DataFrame):
                    subvalue.to_csv(output_dir / f"{subkey}.csv", index=False)


def simple_bar_html(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    n: int = 15,
    value_as_percent: bool = False,
) -> str:
    rows = df.head(n).copy()
    max_value = rows[value_col].max()
    items = []
    for _, row in rows.iterrows():
        width = 0 if max_value == 0 else float(row[value_col]) / float(max_value) * 100
        items.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{fmt(row[label_col])}</div>
              <div class="bar-track"><div class="bar" style="width:{width:.1f}%"></div></div>
              <div class="bar-value">{fmt(row[value_col], value_as_percent)}</div>
            </div>
            """
        )
    return f"<section><h2>{title}</h2>{''.join(items)}</section>"


def write_html_dashboard(results: dict[str, object], output_dir: Path) -> None:
    summary = results["summary"]
    subject_stats = results["subject_stats"]
    student_bins = results["student_attempt_bins"]
    question_bins = results["question_accuracy_bins"]

    stable_subjects = subject_stats.loc[subject_stats["attempts"] >= 1000].copy()
    stable_subjects["label"] = stable_subjects["Name"].fillna(stable_subjects["SubjectId"].astype(str))
    hardest_subjects = stable_subjects.sort_values(["accuracy", "attempts"], ascending=[True, False])

    student_bins_plot = student_bins.rename(columns={"bin": "label", "count": "students"})
    question_bins_plot = question_bins.rename(columns={"bin": "label", "count": "questions"})

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MistakeCoach Eedi EDA</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ margin-top: 32px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 24px 0; }}
    .card {{ border: 1px solid #d9dee7; border-radius: 8px; padding: 14px 16px; background: #fbfcfe; }}
    .metric {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(160px, 260px) 1fr 90px; gap: 12px; align-items: center; margin: 8px 0; }}
    .bar-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ background: #eef2f7; height: 14px; border-radius: 999px; overflow: hidden; }}
    .bar {{ background: #3867d6; height: 100%; }}
    .bar-value {{ text-align: right; font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body>
  <h1>MistakeCoach Eedi Exploratory Data Analysis</h1>
  <p>Automatically generated core EDA dashboard. Full tables are available in the CSV outputs in the same directory.</p>
  <div class="cards">
    <div class="card">Answer records<div class="metric">{fmt(summary["n_rows"])}</div></div>
    <div class="card">Students<div class="metric">{fmt(summary["n_students"])}</div></div>
    <div class="card">Questions<div class="metric">{fmt(summary["n_questions"])}</div></div>
    <div class="card">Overall accuracy<div class="metric">{pct(summary["overall_accuracy"])}</div></div>
  </div>
  {simple_bar_html(hardest_subjects, "label", "difficulty", "Hardest subjects: difficulty = 1 - accuracy", value_as_percent=True)}
  {simple_bar_html(student_bins_plot, "label", "students", "Student attempt count distribution")}
  {simple_bar_html(question_bins_plot, "label", "questions", "Question accuracy distribution")}
</body>
</html>
"""
    with open(output_dir / "eda_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)


def write_report(results: dict[str, object], report_path: Path, output_dir: Path) -> None:
    summary = results["summary"]
    answer_meta_summary = results.get("answer_metadata_summary", {})

    question_stats = results["question_stats"]
    subject_stats = results["subject_stats"]
    answer_summary = results["answer_summary"]
    student_missing = results["student_metadata_missing"]
    answer_missing = results.get("answer_metadata_missing", pd.DataFrame())
    gender = results["accuracy_by_gender"]
    premium = results["accuracy_by_premium"]
    wrong_patterns = results["question_wrong_patterns"]

    stable_hard_questions = question_stats.loc[question_stats["attempts"] >= 100].sort_values(
        ["accuracy", "attempts"], ascending=[True, False]
    )
    stable_subjects = subject_stats.loc[subject_stats["attempts"] >= 1000].copy()
    stable_subjects["subject_label"] = stable_subjects["Name"].fillna(stable_subjects["SubjectId"].astype(str))
    hardest_subjects = stable_subjects.sort_values(["accuracy", "attempts"], ascending=[True, False])
    easiest_subjects = stable_subjects.sort_values(["accuracy", "attempts"], ascending=[False, False])

    report = f"""# MistakeCoach Eedi Exploratory Data Analysis Report

## 1. Data Overview

This analysis uses the Eedi / NeurIPS Education Challenge Task 1/2 training data as the main data source. The core prediction target is `IsCorrect`, which indicates whether a student answered a math diagnostic question correctly.

- Answer records: {fmt(summary["n_rows"])}
- Students: {fmt(summary["n_students"])}
- Questions: {fmt(summary["n_questions"])}
- Correct answers: {fmt(summary["n_correct"])}
- Overall accuracy: {pct(summary["overall_accuracy"])}

Answer metadata records: {fmt(answer_meta_summary.get("n_rows", np.nan))}. The answer timestamp range is approximately {answer_meta_summary.get("date_min", "")} to {answer_meta_summary.get("date_max", "")}.

## 2. Student Activity

The distribution of student answer counts is highly imbalanced. The median number of attempts per student is {fmt(summary["student_attempt_median"])}, with an interquartile range from {fmt(summary["student_attempt_q1"])} to {fmt(summary["student_attempt_q3"])}. This means the model should not rely only on highly active students, or it may overestimate generalization.

## 3. Question Difficulty

The median number of attempts per question is {fmt(summary["question_attempt_median"])}, with an interquartile range from {fmt(summary["question_attempt_q1"])} to {fmt(summary["question_attempt_q3"])}. The median question accuracy is {pct(summary["question_accuracy_median"])}, and the middle 50% of question accuracies fall between {pct(summary["question_accuracy_q1"])} and {pct(summary["question_accuracy_q3"])}.

- Share of questions below 30% accuracy: {pct(summary["questions_below_30_accuracy_share"])}
- Share of questions above 70% accuracy: {pct(summary["questions_above_70_accuracy_share"])}

The dataset contains both easier questions and many diagnostic questions with strong discrimination, making it suitable for difficulty estimation and personalized recommendation.

### Examples of the Hardest Questions (attempts >= 100)

{markdown_table(stable_hard_questions, ["QuestionId", "attempts", "accuracy", "difficulty"], ["QuestionId", "Attempts", "Accuracy", "Difficulty"], 10, {"accuracy", "difficulty"})}

## 4. Subject Performance

The `SubjectId` field in the question metadata is a list of subject tags, so the same question may be counted under multiple subjects. The subject statistics below aggregate across all subject tags linked to each question.

### Lowest-Accuracy Subjects (attempts >= 1000)

{markdown_table(hardest_subjects, ["SubjectId", "subject_label", "Level", "n_questions", "attempts", "accuracy"], ["SubjectId", "Name", "Level", "Questions", "Attempts", "Accuracy"], 10, {"accuracy"})}

### Highest-Accuracy Subjects (attempts >= 1000)

{markdown_table(easiest_subjects, ["SubjectId", "subject_label", "Level", "n_questions", "attempts", "accuracy"], ["SubjectId", "Name", "Level", "Questions", "Attempts", "Accuracy"], 10, {"accuracy"})}

## 5. Answer Choices and Error Patterns

The overall answer option distribution is shown below. This table helps identify whether students favor certain options and whether wrong answers are concentrated on specific distractors.

{markdown_table(answer_summary, ["answer_value", "selected_count", "selected_share", "wrong_selected_count", "wrong_selected_share"], ["Answer", "Selected", "Selected Share", "Wrong Selected", "Wrong Share"], 10, {"selected_share", "wrong_selected_share"})}

### Most Common Question-Level Wrong Options

{markdown_table(wrong_patterns, ["QuestionId", "AnswerValue", "wrong_count", "top_wrong_share_of_wrong", "accuracy"], ["QuestionId", "Top Wrong Answer", "Wrong Count", "Share of Wrong", "Question Accuracy"], 10, {"top_wrong_share_of_wrong", "accuracy"})}

## 6. Student Metadata

Student metadata can be used as supplementary features, but missing values need careful handling. In particular, `DateOfBirth` and `PremiumPupil` have substantial missingness, so modeling should add missing indicators and compare model performance with and without these variables.

### Student Metadata Missing Rates

{markdown_table(student_missing, ["field", "missing_count", "missing_share"], ["Field", "Missing Count", "Missing Share"], 10, {"missing_share"})}

### Summary by Gender

{markdown_table(gender, ["Gender", "students", "attempts", "accuracy"], ["Gender", "Students", "Attempts", "Accuracy"], 10, {"accuracy"})}

### Summary by PremiumPupil

{markdown_table(premium, ["PremiumPupil", "students", "attempts", "accuracy"], ["PremiumPupil", "Students", "Attempts", "Accuracy"], 10, {"accuracy"})}

## 7. Answer Metadata Missing Rates

{markdown_table(answer_missing, ["field", "missing_count", "missing_share"], ["Field", "Missing Count", "Missing Share"], 10, {"missing_share"})}

## 8. Modeling Implications

1. `student_prior_accuracy`, `prior_attempts`, and `student_subject_accuracy` should be core student history features.
2. `question_prior_accuracy` and question attempt counts can be used as question difficulty features.
3. The `SubjectId` hierarchy is useful for subject-level feature engineering and for dashboard views of student weaknesses.
4. Student and question activity are highly imbalanced, so final model evaluation should use a time split or student-group split to avoid leakage.
5. Wrong-answer concentration can support misconception analysis: if most wrong answers for a question concentrate on one option, that option may represent a stable misconception.

## 9. Output Files

The main files generated by this analysis are saved in `{output_dir}`:

- `summary.json`
- `student_stats.csv`
- `question_stats.csv`
- `subject_stats.csv`
- `answer_summary.csv`
- `question_wrong_patterns.csv`
- `student_metadata_missing.csv`
- `answer_metadata_missing.csv`
- `eda_dashboard.html`

The next step is to build a machine learning dataset from these outputs and train baseline, logistic regression, random forest, and gradient boosting models.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Eedi data for MistakeCoach.")
    parser.add_argument("--task", choices=["1_2", "3_4"], default="1_2")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()

    train_path = EEDI_DATA_DIR / "train_data" / f"train_task_{args.task}.csv"
    question_meta_path = EEDI_DATA_DIR / "metadata" / f"question_metadata_task_{args.task}.csv"
    student_meta_path = EEDI_DATA_DIR / "metadata" / f"student_metadata_task_{args.task}.csv"
    answer_meta_path = EEDI_DATA_DIR / "metadata" / f"answer_metadata_task_{args.task}.csv"
    subject_meta_path = EEDI_DATA_DIR / "metadata" / "subject_metadata.csv"

    train_results = analyze_train(train_path, args.chunksize, args.max_chunks)
    subject_stats, question_subject_counts = analyze_question_subjects(
        train_results["question_stats"],
        question_meta_path,
        subject_meta_path,
    )
    student_meta_results = analyze_student_metadata(train_results["student_stats"], student_meta_path)
    answer_meta_results = analyze_answer_metadata(answer_meta_path, args.chunksize, args.max_chunks)

    results: dict[str, object] = {
        **train_results,
        "subject_stats": subject_stats,
        "question_subject_counts": question_subject_counts,
        **student_meta_results,
        **answer_meta_results,
    }

    write_outputs(results, PROCESSED_DIR)
    write_html_dashboard(results, PROCESSED_DIR)
    write_report(results, REPORT_PATH, PROCESSED_DIR)

    print(f"EDA outputs saved to: {PROCESSED_DIR}", flush=True)
    print(f"EDA report saved to: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()

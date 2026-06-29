from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import RAW_DATA_DIR


def load_questions(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else RAW_DATA_DIR / "questions.csv"
    df = pd.read_csv(path)

    required = {
        "question_id",
        "skill_id",
        "question_text",
        "answer",
        "answer_type",
        "difficulty",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"questions.csv is missing required columns: {missing}")

    df["question_id"] = df["question_id"].astype(str)
    df["skill_id"] = df["skill_id"].astype(str)
    df["answer"] = df["answer"].astype(str)
    return df


def load_interactions(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else RAW_DATA_DIR / "interactions.csv"
    df = pd.read_csv(path)

    required = {
        "student_id",
        "question_id",
        "skill_id",
        "student_answer",
        "correct",
        "hint_used",
        "timestamp",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"interactions.csv is missing required columns: {missing}")

    df["student_id"] = df["student_id"].astype(str)
    df["question_id"] = df["question_id"].astype(str)
    df["skill_id"] = df["skill_id"].astype(str)
    df["correct"] = df["correct"].astype(int)
    df["hint_used"] = df["hint_used"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def save_interaction(
    interaction: dict,
    path: str | Path | None = None,
) -> None:
    path = Path(path) if path else RAW_DATA_DIR / "interactions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([interaction])
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old, new_row], ignore_index=True)
    else:
        out = new_row

    out.to_csv(path, index=False)

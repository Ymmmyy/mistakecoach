from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import plotly.express as px


def summarize_interactions(interactions: pd.DataFrame) -> dict:
    return {
        "n_interactions": int(len(interactions)),
        "n_students": int(interactions["student_id"].nunique()),
        "n_questions": int(interactions["question_id"].nunique()),
        "n_skills": int(interactions["skill_id"].nunique()),
        "overall_accuracy": float(interactions["correct"].mean()),
        "avg_hint_used": float(interactions["hint_used"].mean()),
    }


def skill_summary(interactions: pd.DataFrame) -> pd.DataFrame:
    return (
        interactions.groupby("skill_id")
        .agg(
            attempts=("correct", "size"),
            accuracy=("correct", "mean"),
            avg_hint_used=("hint_used", "mean"),
        )
        .reset_index()
        .sort_values("accuracy")
    )


def question_summary(interactions: pd.DataFrame) -> pd.DataFrame:
    return (
        interactions.groupby(["question_id", "skill_id"])
        .agg(
            attempts=("correct", "size"),
            accuracy=("correct", "mean"),
            avg_hint_used=("hint_used", "mean"),
        )
        .reset_index()
        .sort_values("accuracy")
    )


def save_eda_outputs(
    interactions: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_interactions(interactions)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    skill_df = skill_summary(interactions)
    skill_df.to_csv(output_dir / "skill_summary.csv", index=False)

    question_df = question_summary(interactions)
    question_df.to_csv(output_dir / "question_summary.csv", index=False)

    fig1 = px.bar(
        skill_df,
        x="skill_id",
        y="accuracy",
        title="Accuracy by Skill",
    )
    fig1.write_html(output_dir / "accuracy_by_skill.html")

    fig2 = px.scatter(
        skill_df,
        x="avg_hint_used",
        y="accuracy",
        size="attempts",
        hover_name="skill_id",
        title="Hint Usage vs Accuracy by Skill",
    )
    fig2.write_html(output_dir / "hint_usage_vs_accuracy.html")

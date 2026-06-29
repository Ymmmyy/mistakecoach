from __future__ import annotations

import pandas as pd


def compute_skill_mastery(
    interactions: pd.DataFrame,
    student_id: str,
    min_attempts: int = 1,
) -> pd.DataFrame:
    """
    Compute transparent skill-level mastery.

    Formula:
    mastery = 0.65 * accuracy + 0.25 * recent_accuracy + 0.10 * independence

    where independence = 1 - normalized_hint_usage.

    This is intentionally interpretable. You can compare it with logistic regression.
    """
    df = interactions[interactions["student_id"].astype(str) == str(student_id)].copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "skill_id",
                "attempts",
                "accuracy",
                "recent_accuracy",
                "avg_hint_used",
                "mastery",
                "status",
            ]
        )

    df = df.sort_values("timestamp")

    rows = []
    for skill_id, group in df.groupby("skill_id"):
        attempts = len(group)
        if attempts < min_attempts:
            continue

        accuracy = group["correct"].mean()
        recent_accuracy = group.tail(min(5, len(group)))["correct"].mean()
        avg_hint = group["hint_used"].mean()
        independence = max(0.0, 1.0 - min(avg_hint, 3) / 3)

        mastery = 0.65 * accuracy + 0.25 * recent_accuracy + 0.10 * independence

        if mastery >= 0.80:
            status = "strong"
        elif mastery >= 0.60:
            status = "developing"
        else:
            status = "weak"

        rows.append(
            {
                "skill_id": skill_id,
                "attempts": attempts,
                "accuracy": round(accuracy, 3),
                "recent_accuracy": round(recent_accuracy, 3),
                "avg_hint_used": round(avg_hint, 3),
                "mastery": round(mastery, 3),
                "status": status,
            }
        )

    return pd.DataFrame(rows).sort_values("mastery")

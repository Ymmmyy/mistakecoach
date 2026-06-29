from __future__ import annotations

import pandas as pd


def recommend_next_question(
    questions: pd.DataFrame,
    interactions: pd.DataFrame,
    student_id: str,
    mastery_df: pd.DataFrame,
) -> pd.Series:
    """
    Recommend a next question.

    Logic:
    1. Prefer weakest skill with available unseen questions.
    2. Otherwise recommend a lower-difficulty unseen question.
    3. Otherwise repeat from weak skill.
    """
    student_history = interactions[
        interactions["student_id"].astype(str) == str(student_id)
    ]
    seen_question_ids = set(student_history["question_id"].astype(str))

    if not mastery_df.empty:
        weak_skills = list(mastery_df.sort_values("mastery")["skill_id"])
    else:
        weak_skills = list(questions["skill_id"].drop_duplicates())

    for skill in weak_skills:
        candidates = questions[
            (questions["skill_id"].astype(str) == str(skill))
            & (~questions["question_id"].astype(str).isin(seen_question_ids))
        ].copy()
        if not candidates.empty:
            return candidates.sort_values("difficulty").iloc[0]

    unseen = questions[~questions["question_id"].astype(str).isin(seen_question_ids)]
    if not unseen.empty:
        return unseen.sort_values("difficulty").iloc[0]

    if weak_skills:
        repeated = questions[questions["skill_id"].astype(str) == str(weak_skills[0])]
        if not repeated.empty:
            return repeated.sort_values("difficulty").iloc[0]

    return questions.sample(1).iloc[0]

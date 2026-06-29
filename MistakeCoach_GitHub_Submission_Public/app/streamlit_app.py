from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DEFAULT_STUDENT_ID
from src.data.load_data import load_questions, load_interactions, save_interaction
from src.tutor.tutor_engine import TutorEngine
from src.tutor.mastery import compute_skill_mastery
from src.tutor.recommender import recommend_next_question


st.set_page_config(
    page_title="MistakeCoach AI Tutor",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 MistakeCoach")
st.caption("A misconception-aware AI tutor for personalized math practice")

student_id = st.sidebar.text_input("Student ID", value=DEFAULT_STUDENT_ID)
page = st.sidebar.radio(
    "Navigation",
    ["Practice", "Mastery Dashboard", "Data Explorer", "About"],
)

questions = load_questions()
interactions = load_interactions()

if "current_question_id" not in st.session_state:
    mastery_df_init = compute_skill_mastery(interactions, student_id)
    q = recommend_next_question(questions, interactions, student_id, mastery_df_init)
    st.session_state.current_question_id = q["question_id"]

if "hint_level" not in st.session_state:
    st.session_state.hint_level = 1


def get_current_question() -> pd.Series:
    qdf = questions[
        questions["question_id"].astype(str)
        == str(st.session_state.current_question_id)
    ]
    if qdf.empty:
        return questions.iloc[0]
    return qdf.iloc[0]


def move_to_next_question():
    fresh_interactions = load_interactions()
    mastery_df = compute_skill_mastery(fresh_interactions, student_id)
    next_q = recommend_next_question(questions, fresh_interactions, student_id, mastery_df)
    st.session_state.current_question_id = next_q["question_id"]
    st.session_state.hint_level = 1


if page == "Practice":
    col1, col2 = st.columns([2, 1])

    with col1:
        q = get_current_question()
        st.subheader("Practice Problem")
        st.markdown(f"**Skill:** `{q['skill_id']}`")
        st.markdown(f"**Difficulty:** {q['difficulty']}")
        st.info(q["question_text"])

        student_answer = st.text_input("Your answer or attempt:")

        c1, c2, c3 = st.columns(3)
        submit = c1.button("Submit answer", type="primary")
        hint = c2.button("Get another hint")
        next_problem = c3.button("Next problem")

        if hint:
            st.session_state.hint_level += 1

        if next_problem:
            move_to_next_question()
            st.rerun()

        if submit and student_answer.strip():
            engine = TutorEngine()
            turn = engine.respond(
                question_text=q["question_text"],
                correct_answer=q["answer"],
                student_answer=student_answer,
                answer_type=q["answer_type"],
                skill_id=q["skill_id"],
                hint_level=st.session_state.hint_level,
            )

            st.markdown("### Tutor Feedback")
            if turn.check.correct:
                st.success(turn.feedback.text)
            else:
                st.warning(turn.feedback.text)

            with st.expander("System diagnostics"):
                st.write(
                    {
                        "correct": turn.check.correct,
                        "misconception": turn.misconception.label,
                        "misconception_confidence": turn.misconception.confidence,
                        "used_llm": turn.feedback.used_llm,
                        "model": turn.feedback.model,
                        "hint_level": turn.hint_level,
                    }
                )

            save_interaction(
                {
                    "student_id": student_id,
                    "question_id": q["question_id"],
                    "skill_id": q["skill_id"],
                    "student_answer": student_answer,
                    "correct": int(turn.check.correct),
                    "hint_used": int(st.session_state.hint_level - 1),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            if turn.check.correct:
                st.session_state.hint_level = 1

    with col2:
        st.subheader("Current Mastery")
        fresh_interactions = load_interactions()
        mastery_df = compute_skill_mastery(fresh_interactions, student_id)
        if mastery_df.empty:
            st.write("No attempts yet.")
        else:
            st.dataframe(mastery_df, use_container_width=True)
            fig = px.bar(
                mastery_df,
                x="skill_id",
                y="mastery",
                color="status",
                title="Mastery by Skill",
                range_y=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Mastery Dashboard":
    st.subheader("Mastery Dashboard")
    fresh_interactions = load_interactions()
    student_df = fresh_interactions[
        fresh_interactions["student_id"].astype(str) == str(student_id)
    ]

    if student_df.empty:
        st.info("No data for this student yet.")
    else:
        mastery_df = compute_skill_mastery(fresh_interactions, student_id)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Attempts", len(student_df))
        c2.metric("Accuracy", f"{student_df['correct'].mean():.1%}")
        c3.metric("Avg Hints", f"{student_df['hint_used'].mean():.2f}")
        c4.metric("Skills Practiced", student_df["skill_id"].nunique())

        st.dataframe(mastery_df, use_container_width=True)

        fig = px.bar(
            mastery_df,
            x="skill_id",
            y="mastery",
            color="status",
            title="Skill Mastery",
            range_y=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

        trend = (
            student_df.sort_values("timestamp")
            .assign(attempt_number=lambda d: range(1, len(d) + 1))
        )
        fig2 = px.line(
            trend,
            x="attempt_number",
            y="correct",
            color="skill_id",
            markers=True,
            title="Correctness Over Time",
        )
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Data Explorer":
    st.subheader("Dataset Explorer")

    tab1, tab2, tab3 = st.tabs(["Questions", "Interactions", "Skill Analytics"])

    with tab1:
        st.dataframe(questions, use_container_width=True)

    with tab2:
        st.dataframe(interactions, use_container_width=True)

    with tab3:
        skill_stats = (
            interactions.groupby("skill_id")
            .agg(
                attempts=("correct", "size"),
                accuracy=("correct", "mean"),
                avg_hint_used=("hint_used", "mean"),
            )
            .reset_index()
        )
        st.dataframe(skill_stats, use_container_width=True)

        fig = px.scatter(
            skill_stats,
            x="avg_hint_used",
            y="accuracy",
            size="attempts",
            hover_name="skill_id",
            title="Hint Usage vs Accuracy",
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "About":
    st.markdown(
        """
        ## What this product demonstrates

        MistakeCoach is not a general chatbot. It is a structured tutoring workflow:

        1. Selects a practice problem.
        2. Checks the student's answer.
        3. Diagnoses likely misconception.
        4. Generates scaffolded hints.
        5. Updates skill mastery.
        6. Recommends the next problem.
        7. Visualizes student progress.

        For a full course project, replace the demo data with a larger public education dataset.
        """
    )

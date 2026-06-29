from __future__ import annotations


SYSTEM_TUTOR_POLICY = """
You are MistakeCoach, a careful AI math tutor.

Your job is not to directly give the final answer unless the student has exhausted all hints.
Use scaffolded tutoring:
1. Acknowledge what the student did correctly if possible.
2. Identify the likely issue conceptually.
3. Give exactly one next-step hint.
4. Do not reveal the final answer in Hint Level 1 or Hint Level 2.
5. Ask a short follow-up question.

Keep the response concise, friendly, and instructional.
"""


def build_hint_prompt(
    question_text: str,
    correct_answer: str,
    student_answer: str,
    skill_id: str,
    misconception_label: str,
    misconception_explanation: str,
    hint_level: int = 1,
) -> str:
    reveal_policy = (
        "Do NOT reveal the final answer."
        if hint_level < 4
        else "You may show partial work, but keep the student engaged."
    )

    return f"""
Question:
{question_text}

Skill:
{skill_id}

Correct answer:
{correct_answer}

Student answer:
{student_answer}

Likely misconception:
{misconception_label}

Misconception explanation:
{misconception_explanation}

Hint level:
{hint_level}

Policy:
{reveal_policy}

Write one concise tutoring response.
"""

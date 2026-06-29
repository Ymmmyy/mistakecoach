from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MisconceptionResult:
    label: str
    confidence: float
    explanation: str


def diagnose_misconception(
    question_text: str,
    student_answer: str,
    correct_answer: str,
    skill_id: str,
) -> MisconceptionResult:
    """
    Lightweight rule-based misconception diagnosis.

    This is intentionally simple and transparent for a course project.
    You can extend it with more rules or replace it with an ML classifier.
    """
    q = str(question_text).lower()
    a = str(student_answer).lower()
    skill = str(skill_id).lower()

    # Common denominator mistake in mean/average problems.
    if "mean" in q or "average" in q:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", q)
        if len(numbers) >= 2:
            return MisconceptionResult(
                label="possible_denominator_error",
                confidence=0.75,
                explanation="The student may have used the wrong number of values when dividing.",
            )

    # Fraction misconception.
    if "fraction" in skill or "/" in q:
        if "/" in a or True:
            return MisconceptionResult(
                label="possible_fraction_operation_error",
                confidence=0.65,
                explanation="The student may be applying numerator/denominator operations incorrectly.",
            )

    # Equation solving misconception.
    if "equation" in skill or "solve for x" in q:
        return MisconceptionResult(
            label="possible_inverse_operation_error",
            confidence=0.60,
            explanation="The student may have made an inverse-operation or algebraic manipulation error.",
        )

    # Probability misconception.
    if "probability" in skill or "probability" in q:
        return MisconceptionResult(
            label="possible_probability_reasoning_error",
            confidence=0.60,
            explanation="The student may have confused favorable outcomes with total outcomes.",
        )

    return MisconceptionResult(
        label="unknown_misconception",
        confidence=0.30,
        explanation="No specific rule matched. Use a general conceptual hint.",
    )

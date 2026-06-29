from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class CheckResult:
    correct: bool
    normalized_student_answer: str
    normalized_correct_answer: str
    reason: str


def _normalize_text(value: str) -> str:
    return str(value).strip().lower()


def _extract_number(value: str) -> float | None:
    text = str(value).strip()
    match = re.search(r"-?\d+(\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def check_answer(
    student_answer: str,
    correct_answer: str,
    answer_type: str = "numeric",
    tolerance: float = 1e-2,
) -> CheckResult:
    """
    Deterministic answer checker.

    Supports:
    - numeric: compares first numeric value with tolerance.
    - multiple_choice: compares normalized strings.
    - text: simple normalized exact match.
    """
    answer_type = _normalize_text(answer_type)

    if answer_type == "numeric":
        s_num = _extract_number(student_answer)
        c_num = _extract_number(correct_answer)

        if s_num is None or c_num is None:
            return CheckResult(
                correct=False,
                normalized_student_answer=str(student_answer),
                normalized_correct_answer=str(correct_answer),
                reason="Could not parse numeric answer.",
            )

        is_correct = math.isclose(s_num, c_num, abs_tol=tolerance)
        return CheckResult(
            correct=is_correct,
            normalized_student_answer=str(s_num),
            normalized_correct_answer=str(c_num),
            reason="Numeric comparison with tolerance.",
        )

    if answer_type == "multiple_choice":
        s = _normalize_text(student_answer).replace(".", "").replace(")", "")
        c = _normalize_text(correct_answer).replace(".", "").replace(")", "")
        return CheckResult(
            correct=s == c,
            normalized_student_answer=s,
            normalized_correct_answer=c,
            reason="Multiple-choice exact match.",
        )

    s = _normalize_text(student_answer)
    c = _normalize_text(correct_answer)
    return CheckResult(
        correct=s == c,
        normalized_student_answer=s,
        normalized_correct_answer=c,
        reason="Text exact match.",
    )

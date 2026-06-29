from __future__ import annotations

from dataclasses import dataclass

from src.tutor.answer_checker import check_answer, CheckResult
from src.tutor.misconception import diagnose_misconception, MisconceptionResult
from src.tutor.prompt_templates import build_hint_prompt
from src.tutor.llm_client import LLMClient, LLMResponse


@dataclass
class TutorTurn:
    check: CheckResult
    misconception: MisconceptionResult
    feedback: LLMResponse
    hint_level: int


class TutorEngine:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or LLMClient()

    def respond(
        self,
        question_text: str,
        correct_answer: str,
        student_answer: str,
        answer_type: str,
        skill_id: str,
        hint_level: int = 1,
    ) -> TutorTurn:
        check = check_answer(
            student_answer=student_answer,
            correct_answer=correct_answer,
            answer_type=answer_type,
        )

        if check.correct:
            feedback = LLMResponse(
                text=(
                    "Correct. Nice work. To strengthen your understanding, explain in one sentence "
                    "why this method works."
                ),
                used_llm=False,
                model="deterministic-correct-feedback",
            )
            misconception = MisconceptionResult(
                label="none",
                confidence=1.0,
                explanation="The submitted answer matches the answer key.",
            )
            return TutorTurn(
                check=check,
                misconception=misconception,
                feedback=feedback,
                hint_level=hint_level,
            )

        misconception = diagnose_misconception(
            question_text=question_text,
            student_answer=student_answer,
            correct_answer=correct_answer,
            skill_id=skill_id,
        )

        prompt = build_hint_prompt(
            question_text=question_text,
            correct_answer=correct_answer,
            student_answer=student_answer,
            skill_id=skill_id,
            misconception_label=misconception.label,
            misconception_explanation=misconception.explanation,
            hint_level=hint_level,
        )

        feedback = self.llm_client.generate(prompt)

        return TutorTurn(
            check=check,
            misconception=misconception,
            feedback=feedback,
            hint_level=hint_level,
        )

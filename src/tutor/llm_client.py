from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.tutor.prompt_templates import SYSTEM_TUTOR_POLICY


@dataclass
class LLMResponse:
    text: str
    used_llm: bool
    model: str


class LLMClient:
    """
    Thin wrapper around OpenAI API with a safe fallback.

    The app remains runnable without an API key.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key if api_key is not None else OPENAI_API_KEY
        self.model = model or OPENAI_MODEL

    def generate(self, prompt: str) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                text=self._fallback_hint(prompt),
                used_llm=False,
                model="fallback-template",
            )

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_TUTOR_POLICY},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=220,
            )
            return LLMResponse(
                text=completion.choices[0].message.content.strip(),
                used_llm=True,
                model=self.model,
            )
        except Exception as exc:
            return LLMResponse(
                text=(
                    "I could not reach the LLM service, so here is a fallback hint: "
                    "review the key concept in the problem, identify what quantity is being asked for, "
                    "and check whether your operation matches that quantity."
                ),
                used_llm=False,
                model=f"fallback-after-error: {type(exc).__name__}",
            )

    @staticmethod
    def _fallback_hint(prompt: str) -> str:
        return (
            "Good attempt. Before checking the final answer, focus on the concept being tested. "
            "What quantity is the problem asking for, and what operation connects the given values to that quantity?"
        )

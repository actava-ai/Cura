"""The model under test: an OpenAI-compatible doctor holding a real multi-turn history.

Each turn appends the observation (patient/measurement dialogue, plus the case image for
NEJM) as a user message and samples one assistant turn. Reasoning traces are captured per
turn and stored on the history; by default they are stripped from requests (most servers
ignore or reject unknown fields), but ``preserve_thinking=True`` sends prior turns back
unchanged — including ``reasoning_content`` — for endpoints that support preserved
thinking across turns (e.g. the Cura API's ``thinking: {"keep": "all"}``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cura_eval.client import ChatClient


@dataclass(frozen=True)
class DoctorTurn:
    utterance: str
    reasoning: str | None


def build_doctor_system_prompt(scenario: Any, total_inferences: int, image_request: bool) -> str:
    """Upstream doctor wording with a fixed budget statement (no per-turn count), so the
    conversation keeps one stable system message across the episode."""
    base = (
        "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are "
        "inspecting a patient who you will ask questions in order to understand their disease. You "
        f"are only allowed to ask {total_inferences} questions total before you must make a "
        'decision. You can request test results using the format "REQUEST TEST: [test]". For '
        'example, "REQUEST TEST: Chest_X-Ray". Your dialogue will only be 1-3 sentences in length. '
        'Once you have decided to make a diagnosis please type "DIAGNOSIS READY: [diagnosis here]"'
    )
    if image_request:
        base += ' You may also request medical images related to the disease with "REQUEST IMAGES".'
    return base + (
        f"\n\nBelow is all of the information you have. {scenario.examiner_information()}.\n\n "
        "Remember, you must discover their disease by asking them questions. You are also able "
        "to provide exams."
    )


class Doctor:
    def __init__(
        self,
        system_prompt: str,
        client: ChatClient,
        *,
        max_tokens: int = 8192,
        temperature: float | None = 0.0,
        preserve_thinking: bool = False,
    ):
        self._client = client
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._preserve_thinking = preserve_thinking
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def _request_messages(self) -> list[dict[str, Any]]:
        if self._preserve_thinking:
            return self.messages
        out = []
        for m in self.messages:
            if m.get("role") == "assistant" and "reasoning_content" in m:
                m = {k: v for k, v in m.items() if k != "reasoning_content"}
            out.append(m)
        return out

    async def step(self, observation: str, image_uri: str | None) -> DoctorTurn:
        content: str | list[dict[str, Any]]
        if image_uri:
            parts: list[dict[str, Any]] = []
            if observation:
                parts.append({"type": "text", "text": observation})
            parts.append({"type": "image_url", "image_url": {"url": image_uri}})
            content = parts
        else:
            # Turn 0's observation is ""; some servers reject an empty user message.
            content = observation or "Begin the consultation."
        self.messages.append({"role": "user", "content": content})

        result = await self._client.complete(
            self._request_messages(), max_tokens=self._max_tokens, temperature=self._temperature
        )
        self.messages.append(result.assistant_message())
        return DoctorTurn(utterance=result.content.strip(), reasoning=result.reasoning)
